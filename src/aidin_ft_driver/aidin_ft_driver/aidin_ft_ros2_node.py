#!/usr/bin/env python3
import socket
import struct
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Imu, Temperature


class AidinFTROS2Node(Node):
    def __init__(self):
        super().__init__("aidin_ft_ros2_node")

        # Parameters
        self.declare_parameter("sensor_ip", "192.168.1.199")
        self.declare_parameter("sensor_port", 50000)
        self.declare_parameter("local_ip", "0.0.0.0")
        self.declare_parameter("local_port", 50000)
        self.declare_parameter("frame_id", "aidin_ft_sensor")

        # IMU 단위 변환 옵션
        # 단위가 확실하지 않으면 일단 False로 두고 raw로 publish
        self.declare_parameter("accel_in_g", True)
        self.declare_parameter("gyro_in_deg", True)

        self.sensor_ip = self.get_parameter("sensor_ip").value
        self.sensor_port = int(self.get_parameter("sensor_port").value)
        self.local_ip = self.get_parameter("local_ip").value
        self.local_port = int(self.get_parameter("local_port").value)
        self.frame_id = self.get_parameter("frame_id").value

        self.accel_in_g = bool(self.get_parameter("accel_in_g").value)
        self.gyro_in_deg = bool(self.get_parameter("gyro_in_deg").value)

        self.start_cmd = bytes.fromhex("00 03 02")
        self.stop_cmd = bytes.fromhex("00 03 03")
        self.bias_cmd = bytes.fromhex("00 03 04")

        self.packet_size = 52

        # FT 센서는 최신값이 중요하므로 depth 작게
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.wrench_pub = self.create_publisher(
            WrenchStamped,
            "/aidin_ft/wrench_raw",
            qos
        )

        self.imu_pub = self.create_publisher(
            Imu,
            "/aidin_ft/imu_raw",
            qos
        )

        self.temp_pub = self.create_publisher(
            Temperature,
            "/aidin_ft/temperature",
            qos
        )

        # UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.local_ip, self.local_port))

        # timer loop에서 block되지 않게 짧게 설정
        self.sock.settimeout(0.0)

        self.get_logger().info(
            f"Listening UDP on {self.local_ip}:{self.local_port}"
        )
        self.get_logger().info(
            f"Sending START command to {self.sensor_ip}:{self.sensor_port}"
        )

        self.sock.sendto(self.start_cmd, (self.sensor_ip, self.sensor_port))

        # 1 kHz 근처로 polling
        self.timer = self.create_timer(0.001, self.timer_callback)

        self.packet_count = 0

    def parse_packet(self, data: bytes):
        if len(data) != self.packet_size:
            raise ValueError(f"Invalid packet size: {len(data)} bytes")

        values = struct.unpack(">13f", data)

        return {
            "Fx": values[0],
            "Fy": values[1],
            "Fz": values[2],
            "Tx": values[3],
            "Ty": values[4],
            "Tz": values[5],
            "Ax": values[6],
            "Ay": values[7],
            "Az": values[8],
            "Gx": values[9],
            "Gy": values[10],
            "Gz": values[11],
            "Temp": values[12],
        }

    def timer_callback(self):
        # 한 timer callback에서 socket buffer에 쌓인 packet을 최대한 비움
        while rclpy.ok():
            try:
                data, addr = self.sock.recvfrom(1024)
            except BlockingIOError:
                break
            except socket.timeout:
                break
            except Exception as e:
                self.get_logger().error(f"UDP receive error: {e}")
                break

            if len(data) != self.packet_size:
                self.get_logger().warn(
                    f"Invalid packet size: {len(data)} bytes"
                )
                continue

            stamp = self.get_clock().now().to_msg()
            parsed = self.parse_packet(data)

            self.publish_wrench(parsed, stamp)
            self.publish_imu(parsed, stamp)
            self.publish_temperature(parsed, stamp)

            self.packet_count += 1

            if self.packet_count % 1000 == 0:
                self.get_logger().info(
                    f"Received {self.packet_count} packets"
                )

    def publish_wrench(self, parsed, stamp):
        msg = WrenchStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id

        msg.wrench.force.x = float(parsed["Fx"])
        msg.wrench.force.y = float(parsed["Fy"])
        msg.wrench.force.z = float(parsed["Fz"])

        msg.wrench.torque.x = float(parsed["Tx"])
        msg.wrench.torque.y = float(parsed["Ty"])
        msg.wrench.torque.z = float(parsed["Tz"])

        self.wrench_pub.publish(msg)

    def publish_imu(self, parsed, stamp):
        msg = Imu()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id

        ax = float(parsed["Ax"])
        ay = float(parsed["Ay"])
        az = float(parsed["Az"])

        gx = float(parsed["Gx"])
        gy = float(parsed["Gy"])
        gz = float(parsed["Gz"])

        # 단위 확인 전에는 기본 raw 값 그대로.
        # 나중에 센서 단위가 g, deg/s로 확인되면 parameter로 변환 가능.
        if self.accel_in_g:
            ax *= 9.80665
            ay *= 9.80665
            az *= 9.80665

        if self.gyro_in_deg:
            gx *= math.pi / 180.0
            gy *= math.pi / 180.0
            gz *= math.pi / 180.0

        msg.linear_acceleration.x = ax
        msg.linear_acceleration.y = ay
        msg.linear_acceleration.z = az

        msg.angular_velocity.x = gx
        msg.angular_velocity.y = gy
        msg.angular_velocity.z = gz

        # orientation은 센서 packet에 없으므로 unknown 처리
        msg.orientation_covariance[0] = -1.0

        self.imu_pub.publish(msg)

    def publish_temperature(self, parsed, stamp):
        msg = Temperature()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.temperature = float(parsed["Temp"])

        self.temp_pub.publish(msg)

    def send_bias(self):
        self.get_logger().info("Sending BIAS command")
        self.sock.sendto(self.bias_cmd, (self.sensor_ip, self.sensor_port))

    def destroy_node(self):
        self.get_logger().info("Sending STOP command to FT sensor")
        try:
            self.sock.sendto(self.stop_cmd, (self.sensor_ip, self.sensor_port))
        except Exception:
            pass

        try:
            self.sock.close()
        except Exception:
            pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AidinFTROS2Node()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()