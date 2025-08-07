import os
import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch.logging import get_logger
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.actions import LogInfo
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    logger = get_logger('image_publisher_launch')

    # Declare the launch arguments for image_path and model_path
    image_path_arg = DeclareLaunchArgument(
        'image_path',
        default_value=os.path.join(get_package_share_directory('sample_hr_pose_estimation'), 'input_image.jpg'),
        description='Path to the image file'
    )

    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value= "/opt/model/",
        description='Path to the model file'
    )
        
    # Use LaunchConfiguration to get the values of the arguments
    image_path = LaunchConfiguration('image_path')
    model_path = LaunchConfiguration('model_path')

    logger.info(f'IMAGE_PATH set to: {image_path}')
    logger.info(f'MODEL_PATH set to: {model_path}')
    
    # Get USB_CAMERA_PATH from environment variables
    logger = get_logger('usb_cam_launch')
    usb_camera_path = os.environ.get('USB_CAMERA_PATH','/dev/video2')  # Default to /dev/video0 if not set

    
    logger.info(f'USB_CAMERA_PATH set to: {usb_camera_path}')
    

    hr_pose_estimation_node = Node(
        package='sample_hr_pose_estimation',
        executable='sample_hr_pose_estimation',
        output='screen',
        remappings = [
            ("/image_raw", "/camera/color/image_raw"),
        ],
    )

    camera_node = Node(
        package='usb_cam',  # Package name
        executable='usb_cam_node_exe',  # Executable name
        name='usb_cam_node',  # Node name (optional)
        output='screen',  # Output logs to terminal
        parameters=[
            {'video_device': usb_camera_path},  # Fetch USB_CAMERA_PATH from environment
            {'pixel_format': 'mjpeg2rgb'},
            {'image_width': 640},
            {'image_height': 480},
            {'framerate': 10.0},
            {'queue_size': 1000},
        ],
        remappings = [
            ("/image_raw", "/camera/color/image_raw"),
        ],
    )
        
    nn_inference_container = ComposableNodeContainer(
        name="container",
        namespace='',
        package="rclcpp_components",
        executable="component_container",
        output='screen',
        composable_node_descriptions=[
            ComposableNode(
                package = "qrb_ros_nn_inference",
                plugin = "qrb_ros::nn_inference::QrbRosInferenceNode",
                name = "nn_inference_node",
                parameters=[
                    {
                        "backend_option": "",
                        "model_path": PathJoinSubstitution([model_path, "HRNetPose.tflite"])
                    }
                ]
            )
        ]
    )

    return LaunchDescription([
        image_path_arg,
        model_path_arg,
        nn_inference_container,
        hr_pose_estimation_node,
        camera_node,
    ])