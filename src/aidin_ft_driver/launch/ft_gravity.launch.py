from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # 1. AIDIN FT sensor Ethernet node
        Node(
            package='aidin_ft_driver',
            executable='aidin_ft_ros2_node',
            name='aidin_ft_ros2_node',
            output='screen',
            parameters=[{
                'sensor_ip': '192.168.1.199',
                'sensor_port': 50000,
                'local_ip': '0.0.0.0',
                'local_port': 50000,
                'frame_id': 'aidin_ft_sensor',

                # IMU unit conversion
                'accel_in_g': True,
                'gyro_in_deg': True,
            }]
        ),

        # 2. iPhone / Record3D ARKit pose publisher
        Node(
            package='aidin_ft_driver',
            executable='record3d_ros2_pub',
            name='record3d_ros2_pub',
            output='screen',
            parameters=[{
                # 네 record3d_ros2_pub.py에서 쓰는 parameter가 있으면 여기에 추가
                # 없으면 비워둬도 됨
            }]
        ),

        # 3. Gravity compensation using ARKit orientation
        Node(
            package='aidin_ft_driver',
            executable='gravity_comp_arkit_node',
            name='gravity_comp_arkit_node',
            output='screen',
            parameters=[{
                # input / output topics
                'wrench_topic': '/aidin_ft/wrench_raw',
                'pose_topic': '/record3d/pose_raw',
                'output_topic': '/aidin_ft/wrench_compensated',
                'gravity_wrench_topic': '/aidin_ft/gravity_wrench',
                'gravity_force_norm_topic': '/aidin_ft/gravity_force_norm',

                # ARKit gravity direction
                'gravity_axis': 'arkit_y_down',

                # tool mass
                'm_tool': 0.532,

                # final frame calibration
                # R = R_y(-11.5 deg) R_z(90 deg) R_x(180 deg)
                'ry_deg': -11.5,
                'rz_deg': 90.0,
                'rx_deg': 180.0,

                # final COM fitting result
                'r_com_x': -0.001287,
                'r_com_y': -0.000590,
                'r_com_z': 0.091595,

                # final force bias 후보
                'force_bias_x': 0.883164,
                'force_bias_y': -0.127451,
                'force_bias_z': 2.879208,

                'torque_bias_x': -0.060092,
                'torque_bias_y': 0.020078,
                'torque_bias_z': -0.004618,

                # 보통 false
                'invert_pose_rotation': False,
            }]
        ),
    ])