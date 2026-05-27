import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'aidin_ft_driver'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='home',
    maintainer_email='jeoug1002@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'aidin_ft_ros2_node = aidin_ft_driver.aidin_ft_ros2_node:main',
            'gravity_comp_arkit_node = aidin_ft_driver.gravity_comp_arkit_node:main',
            'iphone_arkit_udp_node = aidin_ft_driver.iphone_arkit_udp_node:main',
            'record3d_ros2_pub = aidin_ft_driver.record3d_ros2_pub:main',
            'gripper_ros2_node = aidin_ft_driver.gripper_ros2_node:main',
            'data_collector_node = aidin_ft_driver.data_collector_node:main',
        ],
    },
)
