#!/usr/bin/env python3

import os
import sys
import yaml
import time
import traceback

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Int32


class GripperROS2Node(Node):
    def __init__(self):
        super().__init__("gripper_ros2_node")

        # =============================
        # Parameters
        # =============================
        self.declare_parameter("gripper_root", "")
        self.declare_parameter("config_path", "")
        self.declare_parameter("cmd_topic", "/gripper/cmd")
        self.declare_parameter("state_topic", "/gripper/state")
        self.declare_parameter("publish_rate_hz", 20.0)

        self.gripper_root = self.get_parameter("gripper_root").value
        self.config_path = self.get_parameter("config_path").value
        self.cmd_topic = self.get_parameter("cmd_topic").value
        self.state_topic = self.get_parameter("state_topic").value

        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        publish_rate_hz = max(publish_rate_hz, 1.0)
        self.publish_period = 1.0 / publish_rate_hz

        # =============================
        # Path setting
        # =============================
        # gripper_root는 hardwares/ 와 configs/ 가 들어있는 프로젝트 루트 경로
        #
        # 예:
        #   /home/home/LEGATO-Gripper
        #
        # 그 안에:
        #   /home/home/LEGATO-Gripper/hardwares/gripper.py
        #   /home/home/LEGATO-Gripper/configs/gripper.yaml
        #
        # 구조라고 가정함.

        if self.gripper_root == "":
            self.get_logger().warn(
                "gripper_root parameter is empty. "
                "Using current working directory as gripper_root."
            )
            self.gripper_root = os.getcwd()

        if self.config_path == "":
            self.config_path = os.path.join(
                self.gripper_root,
                "configs",
                "gripper.yaml",
            )

        if self.gripper_root not in sys.path:
            sys.path.append(self.gripper_root)

        self.get_logger().info(f"gripper_root: {self.gripper_root}")
        self.get_logger().info(f"config_path: {self.config_path}")

        # =============================
        # Load Gripper
        # =============================
        try:
            from hardwares.gripper import Gripper

            with open(self.config_path, "r") as f:
                params = yaml.safe_load(f)

            self.gripper = Gripper(params, verbose_mode=True)
            self.get_logger().info("Gripper initialized successfully.")

        except Exception as e:
            self.get_logger().error("Failed to initialize gripper.")
            self.get_logger().error(str(e))
            self.get_logger().error(traceback.format_exc())
            raise e

        # =============================
        # Internal state
        # 0 = open
        # 1 = close
        # =============================
        self.gripper_state = 0
        self.last_cmd_time = time.time()

        # =============================
        # QoS
        # =============================
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # =============================
        # ROS interfaces
        # =============================
        self.cmd_sub = self.create_subscription(
            Int32,
            self.cmd_topic,
            self.cmd_callback,
            qos,
        )

        self.state_pub = self.create_publisher(
            Int32,
            self.state_topic,
            qos,
        )

        self.timer = self.create_timer(
            self.publish_period,
            self.publish_state,
        )

        self.get_logger().info("Gripper ROS2 node started.")
        self.get_logger().info("Topic rule:")
        self.get_logger().info("  /gripper/cmd   std_msgs/Int32  0=open, 1=close")
        self.get_logger().info("  /gripper/state std_msgs/Int32  0=open, 1=close")
        self.get_logger().info("")
        self.get_logger().info("Test commands:")
        self.get_logger().info(
            "  ros2 topic pub /gripper/cmd std_msgs/Int32 \"{data: 0}\" -1"
        )
        self.get_logger().info(
            "  ros2 topic pub /gripper/cmd std_msgs/Int32 \"{data: 1}\" -1"
        )

    def cmd_callback(self, msg: Int32):
        cmd = int(msg.data)

        if cmd == 0:
            self.open_gripper()

        elif cmd == 1:
            self.close_gripper()

        else:
            self.get_logger().warn(
                f"Invalid gripper command: {cmd}. Use 0=open, 1=close."
            )

    def open_gripper(self):
        self.get_logger().info("Command received: 0 -> open gripper")

        try:
            self.gripper.open_gripper()
            self.gripper_state = 0
            self.last_cmd_time = time.time()

        except Exception as e:
            self.get_logger().error("Failed to open gripper.")
            self.get_logger().error(str(e))
            self.get_logger().error(traceback.format_exc())

    def close_gripper(self):
        self.get_logger().info("Command received: 1 -> close gripper")

        try:
            self.gripper.close_gripper()
            self.gripper_state = 1
            self.last_cmd_time = time.time()

        except Exception as e:
            self.get_logger().error("Failed to close gripper.")
            self.get_logger().error(str(e))
            self.get_logger().error(traceback.format_exc())

    def publish_state(self):
        """
        /gripper/state로 저장용 상태값 publish.

        data = 0 : open
        data = 1 : close

        지금은 실제 encoder feedback이 아니라
        마지막으로 성공적으로 보낸 명령 기준 state임.
        나중에 encoder나 width feedback을 읽을 수 있으면
        이 부분을 실제 상태 기반으로 바꾸면 됨.
        """
        msg = Int32()
        msg.data = int(self.gripper_state)
        self.state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GripperROS2Node()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()