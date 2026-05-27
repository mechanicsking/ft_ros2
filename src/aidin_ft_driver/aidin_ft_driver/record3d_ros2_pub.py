#!/usr/bin/env python3
"""
iPhone Record3D → ROS2 Publisher

중력보상용:
  /record3d/pose_raw
    geometry_msgs/PoseStamped
    Record3D/ARKit 원본 pose
    gravity compensation node는 이 토픽을 사용

RGB:
  /record3d/rgb/compressed
    sensor_msgs/CompressedImage  (JPEG)
    TrueDepth 카메라는 자동으로 좌우 반전 적용

디버깅용:
  /record3d/pose_relative
    geometry_msgs/PoseStamped
    시작 자세 기준 상대 pose

  /record3d/euler_relative
    geometry_msgs/Vector3Stamped
    시작 자세 기준 roll, pitch, yaw [rad]

실행:
  1. iPhone Record3D 앱에서 USB streaming 시작
  2. source /opt/ros/humble/setup.bash
  3. python3 record3d_ros2_pub.py
"""

import time
import threading
import math
from threading import Event

import cv2
import numpy as np
from record3d import Record3DStream

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from sensor_msgs.msg import CompressedImage


# ─────────────────────────────────────────────
# Quaternion utilities
# quaternion order: x, y, z, w
# ─────────────────────────────────────────────

def normalize_quaternion(qx, qy, qz, qw):
    norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if norm < 1e-12:
        return 0.0, 0.0, 0.0, 1.0
    return qx / norm, qy / norm, qz / norm, qw / norm


def quaternion_inverse(q):
    qx, qy, qz, qw = q
    return -qx, -qy, -qz, qw


def quaternion_multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b

    return (
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    )


def quaternion_to_euler(qx, qy, qz, qw):
    """
    Return roll, pitch, yaw [rad].
    """
    sinr_cosp = 2.0 * (qw*qx + qy*qz)
    cosr_cosp = 1.0 - 2.0 * (qx*qx + qy*qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw*qy - qz*qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw*qz + qx*qy)
    cosy_cosp = 1.0 - 2.0 * (qy*qy + qz*qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


# ─────────────────────────────────────────────
# ROS2 publisher node
# ─────────────────────────────────────────────

class Record3DPublisher(Node):
    def __init__(self):
        super().__init__("record3d_publisher")

        self.declare_parameter("pose_raw_topic", "/record3d/pose_raw")
        self.declare_parameter("pose_relative_topic", "/record3d/pose_relative")
        self.declare_parameter("euler_relative_topic", "/record3d/euler_relative")
        self.declare_parameter("rgb_topic", "/record3d/rgb/compressed")
        self.declare_parameter("frame_id", "record3d_world")
        self.declare_parameter("relative_frame_id", "record3d_initial")
        self.declare_parameter("jpeg_quality", 85)

        self.pose_raw_topic = self.get_parameter("pose_raw_topic").value
        self.pose_relative_topic = self.get_parameter("pose_relative_topic").value
        self.euler_relative_topic = self.get_parameter("euler_relative_topic").value
        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.frame_id = self.get_parameter("frame_id").value
        self.relative_frame_id = self.get_parameter("relative_frame_id").value
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)

        self.pose_raw_pub = self.create_publisher(
            PoseStamped,
            self.pose_raw_topic,
            10,
        )

        self.pose_relative_pub = self.create_publisher(
            PoseStamped,
            self.pose_relative_topic,
            10,
        )

        self.euler_relative_pub = self.create_publisher(
            Vector3Stamped,
            self.euler_relative_topic,
            10,
        )

        self.rgb_pub = self.create_publisher(
            CompressedImage,
            self.rgb_topic,
            10,
        )

        self.initial_pos = None
        self.initial_quat = None
        self._t0 = None

        self.get_logger().info("Record3D ROS2 Publisher started")
        self.get_logger().info(f"pose_raw_topic:        {self.pose_raw_topic}")
        self.get_logger().info(f"pose_relative_topic:   {self.pose_relative_topic}")
        self.get_logger().info(f"euler_relative_topic:  {self.euler_relative_topic}")
        self.get_logger().info(f"rgb_topic:             {self.rgb_topic} (JPEG q={self.jpeg_quality})")

    def publish_rgb(self, rgb: np.ndarray, flip: bool = False):
        """RGB numpy array (H, W, 3) → CompressedImage (JPEG) publish."""
        if flip:
            rgb = cv2.flip(rgb, 1)

        # Record3D는 RGB 순서로 반환 → OpenCV JPEG 인코딩을 위해 BGR 변환
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.format = "jpeg"
        msg.data = buf.tobytes()
        self.rgb_pub.publish(msg)

    def publish_pose(self, tx, ty, tz, qx, qy, qz, qw):
        stamp = self.get_clock().now().to_msg()

        current_pos = (float(tx), float(ty), float(tz))
        current_quat = normalize_quaternion(qx, qy, qz, qw)

        if self.initial_pos is None:
            self.initial_pos = current_pos
            self.initial_quat = current_quat
            self.get_logger().info("Initial Record3D pose saved")

        # ─────────────────────────────
        # 1) Raw pose: 중력보상용
        # 절대 world 원점은 몰라도 됨.
        # gravity direction이 정렬된 원본 quaternion을 그대로 써야 함.
        # ─────────────────────────────
        raw_msg = PoseStamped()
        raw_msg.header.stamp = stamp
        raw_msg.header.frame_id = self.frame_id

        raw_msg.pose.position.x = current_pos[0]
        raw_msg.pose.position.y = current_pos[1]
        raw_msg.pose.position.z = current_pos[2]

        raw_msg.pose.orientation.x = float(current_quat[0])
        raw_msg.pose.orientation.y = float(current_quat[1])
        raw_msg.pose.orientation.z = float(current_quat[2])
        raw_msg.pose.orientation.w = float(current_quat[3])

        self.pose_raw_pub.publish(raw_msg)

        t_sec = stamp.sec + stamp.nanosec * 1e-9
        if self._t0 is None:
            self._t0 = t_sec
        t_sec -= self._t0

        # ─────────────────────────────
        # 2) Relative pose: 디버깅/행동 delta용
        # q_relative = q_initial^-1 * q_current
        # ─────────────────────────────
        relative_quat = quaternion_multiply(
            quaternion_inverse(self.initial_quat),
            current_quat,
        )
        relative_quat = normalize_quaternion(*relative_quat)

        rel_msg = PoseStamped()
        rel_msg.header.stamp = stamp
        rel_msg.header.frame_id = self.relative_frame_id

        rel_msg.pose.position.x = current_pos[0] - self.initial_pos[0]
        rel_msg.pose.position.y = current_pos[1] - self.initial_pos[1]
        rel_msg.pose.position.z = current_pos[2] - self.initial_pos[2]

        rel_msg.pose.orientation.x = float(relative_quat[0])
        rel_msg.pose.orientation.y = float(relative_quat[1])
        rel_msg.pose.orientation.z = float(relative_quat[2])
        rel_msg.pose.orientation.w = float(relative_quat[3])

        self.pose_relative_pub.publish(rel_msg)

        # ─────────────────────────────
        # 3) Relative Euler: 사람이 보기 위한 값
        # 중력보상에는 쓰지 말 것
        # ─────────────────────────────
        roll, pitch, yaw = quaternion_to_euler(*relative_quat)

        euler_msg = Vector3Stamped()
        euler_msg.header.stamp = stamp
        euler_msg.header.frame_id = self.relative_frame_id
        euler_msg.vector.x = float(roll)
        euler_msg.vector.y = float(pitch)
        euler_msg.vector.z = float(yaw)

        self.euler_relative_pub.publish(euler_msg)

        return t_sec, roll, pitch, yaw


# ─────────────────────────────────────────────
# Record3D stream app
# ─────────────────────────────────────────────

class StreamApp:
    def __init__(self, ros_node: Record3DPublisher):
        self.event = Event()
        self.session = None
        self.node = ros_node

        self._last_frame_time = None
        self._fps = 0.0

    def on_new_frame(self):
        now = time.perf_counter()
        if self._last_frame_time is not None:
            dt = now - self._last_frame_time
            if dt > 1e-6:
                self._fps = 1.0 / dt
        self._last_frame_time = now
        self.event.set()

    def on_stream_stopped(self):
        self.node.get_logger().warn("Record3D stream stopped")

    def connect(self, dev_idx=0):
        print("Searching for Record3D devices...")
        devs = Record3DStream.get_connected_devices()
        print(f"{len(devs)} device(s) found")

        for dev in devs:
            print(f"\tID: {dev.product_id}  UDID: {dev.udid}")

        if len(devs) <= dev_idx:
            raise RuntimeError(f"Device #{dev_idx} not found")

        dev = devs[dev_idx]

        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped

        if not self.session.connect(dev):
            raise RuntimeError(
                "Record3D 연결 실패 — iPhone Record3D 앱에서 USB streaming을 먼저 시작해줘."
            )

        print("Connected. Publishing Record3D pose...")

    def run(self):
        TRUEDEPTH = 0  # record3d_viewer.py 참고

        while rclpy.ok():
            if not self.event.wait(timeout=5.0):
                print("Waiting for Record3D frames...")
                continue

            rgb  = self.session.get_rgb_frame()
            pose = self.session.get_camera_pose()
            flip = (self.session.get_device_type() == TRUEDEPTH)

            self.node.publish_rgb(rgb, flip=flip)

            stamp, roll, pitch, yaw = self.node.publish_pose(
                pose.tx, pose.ty, pose.tz,
                pose.qx, pose.qy, pose.qz, pose.qw,
            )

            print(
                f"t={stamp:.3f} | "
                f"FPS: {self._fps:5.1f} | "
                f"pos({pose.tx:+.3f}, {pose.ty:+.3f}, {pose.tz:+.3f}) | "
                f"rel_rpy({roll:+.3f}, {pitch:+.3f}, {yaw:+.3f})",
                end="\r",
            )

            self.event.clear()


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────

def main():
    rclpy.init()

    node = Record3DPublisher()

    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        daemon=True,
    )
    spin_thread.start()

    app = StreamApp(node)
    app.connect(dev_idx=0)

    try:
        app.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()