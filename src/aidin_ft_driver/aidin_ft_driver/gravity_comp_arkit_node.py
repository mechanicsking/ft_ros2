#!/usr/bin/env python3
import math
import numpy as np
from std_msgs.msg import Float64

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import WrenchStamped, PoseStamped


def quat_to_rot(qx, qy, qz, qw):
    """
    Quaternion [x, y, z, w] -> rotation matrix.
    Assumption:
      Pose orientation represents phone/camera frame -> ARKit world frame.
    """
    x, y, z, w = qx, qy, qz, qw
    n = x*x + y*y + z*z + w*w

    if n < 1e-12:
        return np.eye(3)

    s = 2.0 / n

    xx = x*x*s
    yy = y*y*s
    zz = z*z*s
    xy = x*y*s
    xz = x*z*s
    yz = y*z*s
    wx = w*x*s
    wy = w*y*s
    wz = w*z*s

    return np.array([
        [1.0 - yy - zz, xy - wz,       xz + wy],
        [xy + wz,       1.0 - xx - zz, yz - wx],
        [xz - wy,       yz + wx,       1.0 - xx - yy],
    ], dtype=float)


def rot_x(deg):
    th = math.radians(deg)
    c = math.cos(th)
    s = math.sin(th)

    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c,  -s],
        [0.0, s,   c],
    ], dtype=float)
    
def rot_y(deg):
    th = math.radians(deg)
    c = math.cos(th)
    s = math.sin(th)

    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ], dtype=float)


def rot_z(deg):
    th = math.radians(deg)
    c = math.cos(th)
    s = math.sin(th)

    return np.array([
        [c,  -s, 0.0],
        [s,   c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)


class GravityCompARKitNode(Node):
    def __init__(self):
        super().__init__("gravity_comp_arkit_node")

        self.declare_parameter("wrench_topic", "/aidin_ft/wrench_raw")
        self.declare_parameter("pose_topic", "/record3d/pose_raw")
        self.declare_parameter("output_topic", "/aidin_ft/wrench_compensated")
        self.declare_parameter("gravity_wrench_topic", "/aidin_ft/gravity_wrench")
        self.declare_parameter("gravity_force_norm_topic", "/aidin_ft/gravity_force_norm")

        # ARKit gravity alignment: +Y up, -Y down
        self.declare_parameter("gravity_axis", "arkit_y_down")

        # Gripper/tool mass [kg]
        self.declare_parameter("m_tool", 0.532)

        # COM position from FT sensor origin, expressed in FT sensor frame [m]
        # 일단 초기값. 나중에 torque 보면서 보정.
        self.declare_parameter("r_com_x", 0.0)
        self.declare_parameter("r_com_y", 0.0)
        self.declare_parameter("r_com_z", 0.065)

        # iPhone/Record3D frame -> FT sensor frame rotation
        # 네가 말한 R = Ry(-14) Rz(90) Rx(180)
        self.declare_parameter("ry_deg", -14.0)
        self.declare_parameter("rz_deg", 90.0)
        self.declare_parameter("rx_deg", 180.0)

        # 만약 R 방향이 반대면 true로 바꿔서 테스트
        self.declare_parameter("use_R_PS_transpose", False)

        # 만약 Record3D quaternion이 world->phone이면 true로 바꿔서 테스트
        self.declare_parameter("invert_pose_rotation", False)

        # Bias는 처음엔 0으로 둔다.
        # 나중에 static calibration 후 넣으면 됨.
        self.declare_parameter("force_bias_x", 0.0)
        self.declare_parameter("force_bias_y", 0.0)
        self.declare_parameter("force_bias_z", 0.0)

        self.declare_parameter("torque_bias_x", 0.0)
        self.declare_parameter("torque_bias_y", 0.0)
        self.declare_parameter("torque_bias_z", 0.0)

        self.wrench_topic = self.get_parameter("wrench_topic").value
        self.pose_topic = self.get_parameter("pose_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.gravity_wrench_topic = self.get_parameter("gravity_wrench_topic").value
        self.gravity_force_norm_topic = self.get_parameter("gravity_force_norm_topic").value

        self.load_params()

        self.latest_R_WP = None
        
        ft_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.create_subscription(
            PoseStamped,
            self.pose_topic,
            self.pose_cb,
            50,
        )

        self.create_subscription(
            WrenchStamped,
            self.wrench_topic,
            self.wrench_cb,
            ft_qos,
        )
        
        self.comp_pub = self.create_publisher(
            WrenchStamped,
            self.output_topic,
            50,
        )

        self.gravity_pub = self.create_publisher(
            WrenchStamped,
            self.gravity_wrench_topic,
            50,
        )
        
        self.gravity_force_norm_pub = self.create_publisher(
            Float64,
            self.gravity_force_norm_topic,
            50,
        )

        self.get_logger().info("Gravity compensation node started.")
        self.get_logger().info(f"wrench_topic: {self.wrench_topic}")
        self.get_logger().info(f"pose_topic: {self.pose_topic}")
        self.get_logger().info(f"output_topic: {self.output_topic}")
        self.get_logger().info(f"gravity_wrench_topic: {self.gravity_wrench_topic}")
        self.get_logger().info(f"gravity_force_norm_topic: {self.gravity_force_norm_topic}")
        self.print_params()

    def load_params(self):
        self.m_tool = float(self.get_parameter("m_tool").value)

        self.r_com = np.array([
            float(self.get_parameter("r_com_x").value),
            float(self.get_parameter("r_com_y").value),
            float(self.get_parameter("r_com_z").value),
        ], dtype=float)

        ry = float(self.get_parameter("ry_deg").value)
        rz = float(self.get_parameter("rz_deg").value)
        rx = float(self.get_parameter("rx_deg").value)

        R_PS = rot_y(ry) @ rot_z(rz) @ rot_x(rx)

        if bool(self.get_parameter("use_R_PS_transpose").value):
            R_PS = R_PS.T

        self.R_PS = R_PS

        gravity_axis = self.get_parameter("gravity_axis").value
        if gravity_axis == "arkit_y_down":
            self.g_W = np.array([0.0, -9.80665, 0.0], dtype=float)
        elif gravity_axis == "ros_z_down":
            self.g_W = np.array([0.0, 0.0, -9.80665], dtype=float)
        else:
            raise ValueError("gravity_axis must be arkit_y_down or ros_z_down")

        self.invert_pose_rotation = bool(
            self.get_parameter("invert_pose_rotation").value
        )

        self.force_bias = np.array([
            float(self.get_parameter("force_bias_x").value),
            float(self.get_parameter("force_bias_y").value),
            float(self.get_parameter("force_bias_z").value),
        ], dtype=float)

        self.torque_bias = np.array([
            float(self.get_parameter("torque_bias_x").value),
            float(self.get_parameter("torque_bias_y").value),
            float(self.get_parameter("torque_bias_z").value),
        ], dtype=float)

    def print_params(self):
        self.get_logger().info(f"m_tool: {self.m_tool:.6f} kg")
        self.get_logger().info(f"r_com: {self.r_com.tolist()} m")
        self.get_logger().info(f"g_W: {self.g_W.tolist()}")
        self.get_logger().info(f"force_bias: {self.force_bias.tolist()}")
        self.get_logger().info(f"torque_bias: {self.torque_bias.tolist()}")
        self.get_logger().info(f"invert_pose_rotation: {self.invert_pose_rotation}")
        self.get_logger().info(f"R_PS:\n{self.R_PS}")

    def pose_cb(self, msg):
        q = msg.pose.orientation

        R_WP = quat_to_rot(q.x, q.y, q.z, q.w)

        if self.invert_pose_rotation:
            R_WP = R_WP.T

        self.latest_R_WP = R_WP

    def wrench_cb(self, msg):
        if self.latest_R_WP is None:
            return

        F_raw = np.array([
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
        ], dtype=float)

        tau_raw = np.array([
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z,
        ], dtype=float)

        # phone/camera frame -> world frame
        R_WP = self.latest_R_WP

        # FT sensor frame -> world frame
        R_WS = R_WP @ self.R_PS

        # gravity expressed in FT sensor frame
        g_S = R_WS.T @ self.g_W

        # tool gravity force and torque
        F_g = self.m_tool * g_S
        tau_g = np.cross(self.r_com, F_g)
        F_g_norm = float(np.linalg.norm(F_g))

        # compensated wrench
        F_comp = F_raw - self.force_bias - F_g
        tau_comp = tau_raw - self.torque_bias - tau_g

        comp_msg = WrenchStamped()
        comp_msg.header = msg.header
        comp_msg.wrench.force.x = float(F_comp[0])
        comp_msg.wrench.force.y = float(F_comp[1])
        comp_msg.wrench.force.z = float(F_comp[2])
        comp_msg.wrench.torque.x = float(tau_comp[0])
        comp_msg.wrench.torque.y = float(tau_comp[1])
        comp_msg.wrench.torque.z = float(tau_comp[2])

        self.comp_pub.publish(comp_msg)

        grav_msg = WrenchStamped()
        grav_msg.header = msg.header
        grav_msg.wrench.force.x = float(F_g[0])
        grav_msg.wrench.force.y = float(F_g[1])
        grav_msg.wrench.force.z = float(F_g[2])
        grav_msg.wrench.torque.x = float(tau_g[0])
        grav_msg.wrench.torque.y = float(tau_g[1])
        grav_msg.wrench.torque.z = float(tau_g[2])

        self.gravity_pub.publish(grav_msg)
        
        norm_msg = Float64()
        norm_msg.data = F_g_norm
        self.gravity_force_norm_pub.publish(norm_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GravityCompARKitNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()