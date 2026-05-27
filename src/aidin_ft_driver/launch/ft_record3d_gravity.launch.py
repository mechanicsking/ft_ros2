from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ─────────────────────────────────
    # Launch arguments
    # ─────────────────────────────────
    m_tool = LaunchConfiguration('m_tool')
    r_com_x = LaunchConfiguration('r_com_x')
    r_com_y = LaunchConfiguration('r_com_y')
    r_com_z = LaunchConfiguration('r_com_z')

    gripper_root = LaunchConfiguration('gripper_root')
    gripper_config_path = LaunchConfiguration('gripper_config_path')

    return LaunchDescription([
        # ─────────────────────────────
        # Gravity compensation parameters
        # ─────────────────────────────
        DeclareLaunchArgument('m_tool', default_value='0.532'),
        DeclareLaunchArgument('r_com_x', default_value='-0.001287'),
        DeclareLaunchArgument('r_com_y', default_value='-0.000590'),
        DeclareLaunchArgument('r_com_z', default_value='0.091595'),

        # ─────────────────────────────
        # Gripper parameters
        # 네 실제 LEGATO-Gripper 경로에 맞게 필요하면 launch 실행 때 바꿔도 됨.
        # 예:
        # ros2 launch aidin_ft_driver ft_record3d_gravity.launch.py \
        #   gripper_root:=/home/home/LEGATO-Gripper \
        #   gripper_config_path:=/home/home/LEGATO-Gripper/configs/gripper.yaml
        # ─────────────────────────────
        DeclareLaunchArgument(
            'gripper_root',
            default_value='/home/home/LEGATO-Gripper',
        ),
        DeclareLaunchArgument(
            'gripper_config_path',
            default_value='/home/home/LEGATO-Gripper/configs/gripper.yaml',
        ),

        # ─────────────────────────────
        # 1. Aidin FT sensor raw publisher
        # publish:
        #   /aidin_ft/wrench_raw
        #   /aidin_ft/imu_raw
        #   /aidin_ft/temperature
        # ─────────────────────────────
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
                'accel_in_g': True,
                'gyro_in_deg': True,
            }],
        ),

        # ─────────────────────────────
        # 2. iPhone Record3D publisher
        # publish:
        #   /record3d/pose_raw
        #   /record3d/pose_relative
        #   /record3d/euler_relative
        #   /record3d/rgb/compressed
        # ─────────────────────────────
        Node(
            package='aidin_ft_driver',
            executable='record3d_ros2_pub',
            name='record3d_publisher',
            output='screen',
            parameters=[{
                'pose_raw_topic': '/record3d/pose_raw',
                'pose_relative_topic': '/record3d/pose_relative',
                'euler_relative_topic': '/record3d/euler_relative',
                'rgb_topic': '/record3d/rgb/compressed',
                'frame_id': 'record3d_world',
                'relative_frame_id': 'record3d_initial',
                'jpeg_quality': 85,
            }],
        ),

        # ─────────────────────────────
        # 3. Gravity compensation
        # subscribe:
        #   /aidin_ft/wrench_raw
        #   /record3d/pose_raw
        #
        # publish:
        #   /aidin_ft/wrench_compensated
        #   /aidin_ft/gravity_wrench
        #   /aidin_ft/gravity_force_norm
        # ─────────────────────────────
        Node(
            package='aidin_ft_driver',
            executable='gravity_comp_arkit_node',
            name='gravity_comp_arkit_node',
            output='screen',
            parameters=[{
                'wrench_topic': '/aidin_ft/wrench_raw',
                'pose_topic': '/record3d/pose_raw',
                'output_topic': '/aidin_ft/wrench_compensated',
                'gravity_wrench_topic': '/aidin_ft/gravity_wrench',
                'gravity_force_norm_topic': '/aidin_ft/gravity_force_norm',

                'gravity_axis': 'arkit_y_down',

                # Tool mass and COM
                'm_tool': m_tool,
                'r_com_x': r_com_x,
                'r_com_y': r_com_y,
                'r_com_z': r_com_z,

                # iPhone/Record3D frame -> FT sensor frame
                'ry_deg': -11.5,
                'rz_deg': 90.0,
                'rx_deg': 180.0,

                'use_R_PS_transpose': False,
                'invert_pose_rotation': False,

                # bias = 기존 bias + residual 보정값
                'force_bias_x': 0.883164,
                'force_bias_y': -0.127451,
                'force_bias_z': 2.879208,

                'torque_bias_x': -0.060092,
                'torque_bias_y': 0.020078,
                'torque_bias_z': -0.004618,
            }],
        ),

        # ─────────────────────────────
        # 4. Gripper ROS2 node
        # subscribe:
        #   /gripper/cmd
        #
        # publish:
        #   /gripper/state
        #
        # rule:
        #   0 = open
        #   1 = close
        # ─────────────────────────────
        Node(
            package='aidin_ft_driver',
            executable='gripper_ros2_node',
            name='gripper_ros2_node',
            output='screen',
            parameters=[{
                'gripper_root': gripper_root,
                'config_path': gripper_config_path,
                'cmd_topic': '/gripper/cmd',
                'state_topic': '/gripper/state',
                'publish_rate_hz': 20.0,
            }],
        ),
    ])