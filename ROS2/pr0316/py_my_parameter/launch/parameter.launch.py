from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='py_my_parameter',
            executable='py_my_parameter',
            name='parameter_node',
            parameters=[{'my_param': 'hello_param'}]
        )
    ])
