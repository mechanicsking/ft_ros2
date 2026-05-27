#!/usr/bin/env python3
"""
UMI-FT Data Collector Node

저장 토픽:
  /aidin_ft/wrench_compensated      geometry_msgs/WrenchStamped
  /record3d/pose_raw                geometry_msgs/PoseStamped
  /record3d/rgb/compressed          sensor_msgs/CompressedImage
  /gripper/state                    std_msgs/Int32

발행 토픽:
  /gripper/cmd                      std_msgs/Int32
    0 = open
    1 = close

키:
  s : start recording
  e : end and save
  a : abort
  b : gripper toggle, open <-> close
  h : help
  q : quit

저장 파일:
  episode_000000.h5
  episode_000001.h5
  episode_000002.h5
  ...

중간에 빠진 번호가 있으면 그 번호부터 저장.
"""

import os
import sys
import math
import time
import select
import termios
import tty
import threading
import datetime
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import WrenchStamped, PoseStamped
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32

try:
    import h5py
except ImportError:
    h5py = None


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def quat_to_euler(qx, qy, qz, qw):
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def now_wall_string():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class DataCollectorNode(Node):
    def __init__(self):
        super().__init__("data_collector_node")

        # ─────────────────────────────
        # Parameters
        # ─────────────────────────────
        self.declare_parameter("save_dir", os.path.expanduser("~/ft_data"))

        self.declare_parameter("wrench_topic", "/aidin_ft/wrench_compensated")
        self.declare_parameter("pose_topic", "/record3d/pose_raw")
        self.declare_parameter("rgb_topic", "/record3d/rgb/compressed")

        self.declare_parameter("gripper_state_topic", "/gripper/state")
        self.declare_parameter("gripper_cmd_topic", "/gripper/cmd")

        self.declare_parameter("status_interval_sec", 1.0)
        self.declare_parameter("task_name", "umi_ft_task")

        self.save_dir = self.get_parameter("save_dir").value
        self.wrench_topic = self.get_parameter("wrench_topic").value
        self.pose_topic = self.get_parameter("pose_topic").value
        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.gripper_state_topic = self.get_parameter("gripper_state_topic").value
        self.gripper_cmd_topic = self.get_parameter("gripper_cmd_topic").value
        self.task_name = self.get_parameter("task_name").value

        os.makedirs(self.save_dir, exist_ok=True)

        if h5py is None:
            self.get_logger().error(
                "h5py가 없습니다. 설치: python3 -m pip install --user h5py"
            )

        # ─────────────────────────────
        # State
        # ─────────────────────────────
        self._lock = threading.Lock()

        self.recording = False
        self.current_episode_index: Optional[int] = None
        self.episode_start_ros_time: Optional[float] = None
        self.episode_start_wall_time: Optional[str] = None
        self.should_quit = False

        # 최신 토픽 상태 확인용
        self.latest_ft_wall_time: Optional[float] = None
        self.latest_pose_wall_time: Optional[float] = None
        self.latest_rgb_wall_time: Optional[float] = None
        self.latest_gripper_wall_time: Optional[float] = None

        # gripper state
        # 0 = open, 1 = close
        self.latest_gripper_state = 0

        self.clear_buffers()

        # ─────────────────────────────
        # QoS
        # ─────────────────────────────
        ft_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        normal_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        rgb_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        # ─────────────────────────────
        # Subscribers
        # ─────────────────────────────
        self.create_subscription(
            WrenchStamped,
            self.wrench_topic,
            self.ft_callback,
            ft_qos,
        )

        self.create_subscription(
            PoseStamped,
            self.pose_topic,
            self.pose_callback,
            normal_qos,
        )

        self.create_subscription(
            CompressedImage,
            self.rgb_topic,
            self.rgb_callback,
            rgb_qos,
        )

        self.create_subscription(
            Int32,
            self.gripper_state_topic,
            self.gripper_state_callback,
            normal_qos,
        )

        # ─────────────────────────────
        # Publisher
        # ─────────────────────────────
        self.gripper_cmd_pub = self.create_publisher(
            Int32,
            self.gripper_cmd_topic,
            normal_qos,
        )

        # ─────────────────────────────
        # Timer and keyboard
        # ─────────────────────────────
        interval = float(self.get_parameter("status_interval_sec").value)
        self.create_timer(interval, self.status_callback)

        self.keyboard_thread = threading.Thread(
            target=self.keyboard_loop,
            daemon=True,
        )
        self.keyboard_thread.start()

        self.print_startup_info()

    # ─────────────────────────────────
    # Buffers
    # ─────────────────────────────────

    def clear_buffers(self):
        self.ft_timestamps = []
        self.ft_wrenches = []

        self.pose_timestamps = []
        self.pose_position = []
        self.pose_quaternion = []
        self.pose_xyzrpy = []

        self.rgb_timestamps = []
        self.rgb_frames = []

        self.gripper_timestamps = []
        self.gripper_states = []

        self.event_timestamps = []
        self.event_labels = []

    # ─────────────────────────────────
    # Episode index
    # ─────────────────────────────────

    def get_next_episode_index(self) -> int:
        """
        save_dir 안에서 episode_000000.h5부터 확인해서
        비어 있는 가장 작은 번호를 반환.
        """
        idx = 0
        while True:
            path = os.path.join(self.save_dir, f"episode_{idx:06d}.h5")
            if not os.path.exists(path):
                return idx
            idx += 1

    def get_current_filepath(self) -> str:
        if self.current_episode_index is None:
            raise RuntimeError("current_episode_index is None")
        return os.path.join(
            self.save_dir,
            f"episode_{self.current_episode_index:06d}.h5",
        )

    # ─────────────────────────────────
    # Event
    # ─────────────────────────────────

    def add_event_unlocked(self, label: str):
        """
        event = 버튼 누른 시점 기록.
        예: recording_start, gripper_toggle_to_close, recording_end
        """
        if self.episode_start_ros_time is None:
            t_rel = 0.0
        else:
            t_now = self.get_clock().now().nanoseconds * 1e-9
            t_rel = t_now - self.episode_start_ros_time

        self.event_timestamps.append(float(t_rel))
        self.event_labels.append(str(label))

    # ─────────────────────────────────
    # Time conversion
    # ─────────────────────────────────

    def to_episode_time(self, t_abs: float) -> float:
        """
        각 센서 timestamp를 episode 시작 기준 상대시간으로 변환.
        센서마다 Hz가 다르기 때문에 timestamp 배열은 따로 저장한다.
        """
        if self.episode_start_ros_time is None:
            return 0.0
        return float(t_abs - self.episode_start_ros_time)

    # ─────────────────────────────────
    # ROS Callbacks
    # ─────────────────────────────────

    def ft_callback(self, msg: WrenchStamped):
        t_abs = stamp_to_sec(msg.header.stamp)
        self.latest_ft_wall_time = time.time()

        with self._lock:
            if not self.recording:
                return

            self.ft_timestamps.append(self.to_episode_time(t_abs))
            self.ft_wrenches.append([
                msg.wrench.force.x,
                msg.wrench.force.y,
                msg.wrench.force.z,
                msg.wrench.torque.x,
                msg.wrench.torque.y,
                msg.wrench.torque.z,
            ])

    def pose_callback(self, msg: PoseStamped):
        t_abs = stamp_to_sec(msg.header.stamp)
        self.latest_pose_wall_time = time.time()

        p = msg.pose.position
        q = msg.pose.orientation
        roll, pitch, yaw = quat_to_euler(q.x, q.y, q.z, q.w)

        with self._lock:
            if not self.recording:
                return

            self.pose_timestamps.append(self.to_episode_time(t_abs))
            self.pose_position.append([p.x, p.y, p.z])
            self.pose_quaternion.append([q.x, q.y, q.z, q.w])
            self.pose_xyzrpy.append([p.x, p.y, p.z, roll, pitch, yaw])

    def rgb_callback(self, msg: CompressedImage):
        t_abs = stamp_to_sec(msg.header.stamp)
        self.latest_rgb_wall_time = time.time()

        with self._lock:
            if not self.recording:
                return

            self.rgb_timestamps.append(self.to_episode_time(t_abs))
            self.rgb_frames.append(bytes(msg.data))

    def gripper_state_callback(self, msg: Int32):
        t_abs = self.get_clock().now().nanoseconds * 1e-9
        state = int(msg.data)

        self.latest_gripper_wall_time = time.time()
        self.latest_gripper_state = state

        with self._lock:
            if not self.recording:
                return

            self.gripper_timestamps.append(self.to_episode_time(t_abs))
            self.gripper_states.append(state)

    # ─────────────────────────────────
    # Recording control
    # ─────────────────────────────────

    def start_recording(self):
        with self._lock:
            if self.recording:
                self.get_logger().warn("이미 recording 중입니다.")
                return

            self.clear_buffers()

            self.current_episode_index = self.get_next_episode_index()
            self.episode_start_ros_time = self.get_clock().now().nanoseconds * 1e-9
            self.episode_start_wall_time = now_wall_string()
            self.recording = True

            self.add_event_unlocked("recording_start")

            # 시작 시점 gripper 상태를 하나 저장
            self.gripper_timestamps.append(0.0)
            self.gripper_states.append(int(self.latest_gripper_state))

        filepath = self.get_current_filepath()

        self.get_logger().info("")
        self.get_logger().info("====================================")
        self.get_logger().info("RECORDING START")
        self.get_logger().info(f"episode index : {self.current_episode_index:06d}")
        self.get_logger().info(f"save path     : {filepath}")
        self.get_logger().info("====================================")

    def end_and_save(self):
        with self._lock:
            if not self.recording:
                self.get_logger().warn("recording 중이 아닙니다.")
                return

            self.add_event_unlocked("recording_end")
            self.recording = False

        self.save_current_episode()

    def abort_recording(self):
        with self._lock:
            if not self.recording:
                self.get_logger().warn("recording 중이 아닙니다.")
                return

            aborted_idx = self.current_episode_index
            self.add_event_unlocked("recording_abort")
            self.recording = False
            self.clear_buffers()
            self.current_episode_index = None
            self.episode_start_ros_time = None
            self.episode_start_wall_time = None

        self.get_logger().warn(f"episode {aborted_idx:06d} 버림. 저장하지 않았습니다.")

    # ─────────────────────────────────
    # Gripper control
    # ─────────────────────────────────

    def send_gripper_cmd(self, cmd: int):
        """
        /gripper/cmd로 0 또는 1 발행.
          0 = open
          1 = close
        """
        msg = Int32()
        msg.data = int(cmd)
        self.gripper_cmd_pub.publish(msg)

        with self._lock:
            if self.recording:
                label = "gripper_open_cmd" if cmd == 0 else "gripper_close_cmd"
                self.add_event_unlocked(label)

                # 버튼 누른 순간의 state도 저장.
                # 이후 /gripper/state callback에서도 state가 추가로 들어온다.
                t_now = self.get_clock().now().nanoseconds * 1e-9
                self.gripper_timestamps.append(self.to_episode_time(t_now))
                self.gripper_states.append(int(cmd))

        self.get_logger().info(
            f"Published {self.gripper_cmd_topic} = {cmd} "
            f"({'open' if cmd == 0 else 'close'})"
        )

    def toggle_gripper(self):
        """
        b 버튼 하나로 gripper open/close 토글.
        현재 latest_gripper_state 기준:
          0이면 close 명령 1
          1이면 open 명령 0
        """
        if int(self.latest_gripper_state) == 0:
            self.send_gripper_cmd(1)
        else:
            self.send_gripper_cmd(0)

    # ─────────────────────────────────
    # Keyboard
    # ─────────────────────────────────

    def keyboard_loop(self):
        old_settings = termios.tcgetattr(sys.stdin)

        try:
            tty.setcbreak(sys.stdin.fileno())

            while rclpy.ok() and not self.should_quit:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)

                    if key == "s":
                        self.start_recording()

                    elif key == "e":
                        self.end_and_save()

                    elif key == "a":
                        self.abort_recording()

                    elif key == "b":
                        self.toggle_gripper()

                    elif key == "h":
                        self.print_key_help()

                    elif key == "q":
                        self.get_logger().info("Quit requested.")
                        self.should_quit = True

                        if self.recording:
                            self.get_logger().warn(
                                "recording 중 q를 눌렀습니다. 현재 episode를 저장합니다."
                            )
                            self.recording = False
                            self.save_current_episode()

                        rclpy.shutdown()
                        break

        except Exception as e:
            self.get_logger().error(f"keyboard_loop error: {e}")

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    # ─────────────────────────────────
    # Status
    # ─────────────────────────────────

    def topic_age_string(self, latest_time: Optional[float]) -> str:
        if latest_time is None:
            return "NO"
        age = time.time() - latest_time
        if age < 1.0:
            return "OK"
        return f"OLD {age:.1f}s"

    def status_callback(self):
        with self._lock:
            recording = self.recording
            episode_idx = self.current_episode_index

            n_ft = len(self.ft_timestamps)
            n_pose = len(self.pose_timestamps)
            n_rgb = len(self.rgb_timestamps)
            n_grip = len(self.gripper_timestamps)

            if self.episode_start_ros_time is not None:
                elapsed = self.get_clock().now().nanoseconds * 1e-9 - self.episode_start_ros_time
            else:
                elapsed = 0.0

        state_str = "RECORDING" if recording else "IDLE"
        if episode_idx is None:
            ep_str = "none"
        else:
            ep_str = f"{episode_idx:06d}"

        self.get_logger().info(
            f"[{state_str}] ep={ep_str} | "
            f"topic FT={self.topic_age_string(self.latest_ft_wall_time)} "
            f"Pose={self.topic_age_string(self.latest_pose_wall_time)} "
            f"RGB={self.topic_age_string(self.latest_rgb_wall_time)} "
            f"Grip={self.topic_age_string(self.latest_gripper_wall_time)} | "
            f"buf ft={n_ft}, pose={n_pose}, rgb={n_rgb}, grip={n_grip} | "
            f"t={elapsed:.1f}s | gripper_state={self.latest_gripper_state}"
        )

    def print_startup_info(self):
        self.get_logger().info("")
        self.get_logger().info("====================================")
        self.get_logger().info("UMI-FT Data Collector Ready")
        self.get_logger().info("====================================")
        self.get_logger().info(f"save_dir             : {self.save_dir}")
        self.get_logger().info(f"task_name            : {self.task_name}")
        self.get_logger().info(f"wrench_topic         : {self.wrench_topic}")
        self.get_logger().info(f"pose_topic           : {self.pose_topic}")
        self.get_logger().info(f"rgb_topic            : {self.rgb_topic}")
        self.get_logger().info(f"gripper_state_topic  : {self.gripper_state_topic}")
        self.get_logger().info(f"gripper_cmd_topic    : {self.gripper_cmd_topic}")
        self.get_logger().info("====================================")
        self.get_logger().info(
            f"next episode index   : {self.get_next_episode_index():06d}"
        )
        self.print_key_help()

    def print_key_help(self):
        self.get_logger().info("")
        self.get_logger().info("Keys:")
        self.get_logger().info("  s : start recording")
        self.get_logger().info("  e : end and save")
        self.get_logger().info("  a : abort current episode")
        self.get_logger().info("  b : gripper toggle open/close")
        self.get_logger().info("  h : help")
        self.get_logger().info("  q : quit")
        self.get_logger().info("")

    # ─────────────────────────────────
    # Save
    # ─────────────────────────────────

    def save_current_episode(self):
        if h5py is None:
            self.get_logger().error("h5py가 없어서 저장할 수 없습니다.")
            return

        with self._lock:
            if self.current_episode_index is None:
                self.get_logger().warn("current_episode_index가 없습니다. 저장 스킵.")
                return

            ft_ts = np.array(self.ft_timestamps, dtype=np.float64)
            ft_wrench = np.array(self.ft_wrenches, dtype=np.float32)

            pose_ts = np.array(self.pose_timestamps, dtype=np.float64)
            pose_position = np.array(self.pose_position, dtype=np.float32)
            pose_quaternion = np.array(self.pose_quaternion, dtype=np.float32)
            pose_xyzrpy = np.array(self.pose_xyzrpy, dtype=np.float32)

            rgb_ts = np.array(self.rgb_timestamps, dtype=np.float64)
            rgb_frames = list(self.rgb_frames)

            gripper_ts = np.array(self.gripper_timestamps, dtype=np.float64)
            gripper_states = np.array(self.gripper_states, dtype=np.int32)

            event_ts = np.array(self.event_timestamps, dtype=np.float64)
            event_labels = list(self.event_labels)

            episode_idx = int(self.current_episode_index)
            episode_start_wall_time = self.episode_start_wall_time or now_wall_string()

        if len(ft_ts) == 0 and len(pose_ts) == 0 and len(rgb_ts) == 0:
            self.get_logger().warn("수집된 데이터가 없습니다. 저장하지 않습니다.")
            return

        duration_candidates = []
        for arr in [ft_ts, pose_ts, rgb_ts, gripper_ts]:
            if len(arr) > 0:
                duration_candidates.append(float(arr[-1]))

        duration_sec = max(duration_candidates) if duration_candidates else 0.0

        filepath = os.path.join(self.save_dir, f"episode_{episode_idx:06d}.h5")

        if os.path.exists(filepath):
            self.get_logger().error(f"파일이 이미 존재합니다. 덮어쓰지 않습니다: {filepath}")
            return

        self.get_logger().info("")
        self.get_logger().info(f"Saving episode: {filepath}")

        with h5py.File(filepath, "w") as f:
            # attrs
            f.attrs["created_at"] = episode_start_wall_time
            f.attrs["episode_index"] = episode_idx
            f.attrs["task_name"] = str(self.task_name)
            f.attrs["duration_sec"] = float(duration_sec)

            f.attrs["n_ft"] = int(len(ft_ts))
            f.attrs["n_pose"] = int(len(pose_ts))
            f.attrs["n_rgb"] = int(len(rgb_ts))
            f.attrs["n_gripper"] = int(len(gripper_ts))
            f.attrs["n_events"] = int(len(event_ts))

            f.attrs["saved_by"] = "data_collector_node.py"
            f.attrs["timestamp_rule"] = (
                "Each sensor has its own timestamp array because sensors run at different Hz. "
                "All timestamps are relative to episode start."
            )

            f.attrs["wrench_topic"] = self.wrench_topic
            f.attrs["pose_topic"] = self.pose_topic
            f.attrs["rgb_topic"] = self.rgb_topic
            f.attrs["gripper_state_topic"] = self.gripper_state_topic
            f.attrs["gripper_cmd_topic"] = self.gripper_cmd_topic

            # FT
            gft = f.create_group("ft")
            gft.create_dataset("timestamp", data=ft_ts)
            if len(ft_wrench) > 0:
                ds = gft.create_dataset("wrench", data=ft_wrench, compression="lzf")
                ds.attrs["columns"] = ["fx", "fy", "fz", "tx", "ty", "tz"]
                ds.attrs["units"] = ["N", "N", "N", "Nm", "Nm", "Nm"]

            # Pose
            gpose = f.create_group("pose")
            gpose.create_dataset("timestamp", data=pose_ts)

            if len(pose_position) > 0:
                ds = gpose.create_dataset("position", data=pose_position, compression="lzf")
                ds.attrs["columns"] = ["x", "y", "z"]
                ds.attrs["units"] = ["m", "m", "m"]

            if len(pose_quaternion) > 0:
                ds = gpose.create_dataset("quaternion", data=pose_quaternion, compression="lzf")
                ds.attrs["columns"] = ["qx", "qy", "qz", "qw"]

            if len(pose_xyzrpy) > 0:
                ds = gpose.create_dataset("xyzrpy", data=pose_xyzrpy, compression="lzf")
                ds.attrs["columns"] = ["x", "y", "z", "roll", "pitch", "yaw"]
                ds.attrs["units"] = ["m", "m", "m", "rad", "rad", "rad"]

            # RGB
            grgb = f.create_group("rgb")
            grgb.create_dataset("timestamp", data=rgb_ts)
            grgb.attrs["format"] = "jpeg"
            grgb.attrs["source_topic_type"] = "sensor_msgs/CompressedImage"

            if len(rgb_frames) > 0:
                vlen_dt = h5py.vlen_dtype(np.dtype("uint8"))
                ds = grgb.create_dataset(
                    "frames",
                    shape=(len(rgb_frames),),
                    dtype=vlen_dt,
                )

                for i, jpeg in enumerate(rgb_frames):
                    ds[i] = np.frombuffer(jpeg, dtype=np.uint8)

            # Gripper
            ggrip = f.create_group("gripper")
            ggrip.create_dataset("timestamp", data=gripper_ts)
            ggrip.create_dataset("state", data=gripper_states)
            ggrip.attrs["state_rule"] = "0=open, 1=close"

            # Events
            gevt = f.create_group("events")
            gevt.create_dataset("timestamp", data=event_ts)

            str_dt = h5py.string_dtype(encoding="utf-8")
            label_ds = gevt.create_dataset(
                "label",
                shape=(len(event_labels),),
                dtype=str_dt,
            )
            for i, label in enumerate(event_labels):
                label_ds[i] = label

        size_mb = os.path.getsize(filepath) / 1024 / 1024

        self.get_logger().info("====================================")
        self.get_logger().info("SAVE DONE")
        self.get_logger().info(f"file     : {filepath}")
        self.get_logger().info(f"size     : {size_mb:.1f} MB")
        self.get_logger().info(f"duration : {duration_sec:.2f} s")
        self.get_logger().info(f"FT       : {len(ft_ts)}")
        self.get_logger().info(f"Pose     : {len(pose_ts)}")
        self.get_logger().info(f"RGB      : {len(rgb_ts)}")
        self.get_logger().info(f"Gripper  : {len(gripper_ts)}")
        self.get_logger().info(f"Events   : {len(event_ts)}")
        self.get_logger().info("====================================")

        with self._lock:
            self.clear_buffers()
            self.current_episode_index = None
            self.episode_start_ros_time = None
            self.episode_start_wall_time = None


def main(args=None):
    rclpy.init(args=args)
    node = DataCollectorNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt received.")

        if node.recording:
            node.get_logger().warn("Recording 중 종료됨. 현재 episode를 저장합니다.")
            with node._lock:
                node.recording = False
                node.add_event_unlocked("recording_interrupted")
            node.save_current_episode()

    finally:
        try:
            node.destroy_node()
        except Exception:
            pass

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()