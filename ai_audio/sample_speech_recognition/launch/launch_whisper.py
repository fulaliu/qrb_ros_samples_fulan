# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os  # For accessing environment variables
import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.logging import get_logger
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    
    declared_args = [
        DeclareLaunchArgument(
            'encode_model_path',
            #default_value="/home/ubuntu/ASR/qrb_ros_samples_fulan/ai_audio/sample_speech_recognition/whisper_tiny_en-whisperencoder.tflite",
            #default_value="/home/ubuntu/ASR/qrb_ros_samples_fulan/ai_audio/sample_speech_recognition/Whisper-Base_HfWhisperEncoder.bin",
            default_value="/home/ubuntu/ASR/qrb_ros_samples_fulan/ai_audio/sample_speech_recognition/WhisperEncoder.bin",
            description='Path to the encode model file'
        ),
        DeclareLaunchArgument(
            'decode_model_path',
            #default_value="/home/ubuntu/ASR/qrb_ros_samples_fulan/ai_audio/sample_speech_recognition/whisper_tiny_en-whisperdecoder.tflite",
            default_value="/home/ubuntu/ASR/qrb_ros_samples_fulan/ai_audio/sample_speech_recognition/WhisperDecoder.bin",
            description='Path to the decode model file'
        ),
    ]

    encode_model_path = LaunchConfiguration('encode_model_path')
    decode_model_path = LaunchConfiguration('decode_model_path')
    namespace = ""
     
    whisper_preprocess_node = Node(
        package='sample_speech_recognition',
        executable='qrb_ros_whisper_preprocess',
        namespace=namespace,
        output='screen',
    )
    
    nn_encode_inference_node = ComposableNode(
    package = "qrb_ros_nn_inference",
    namespace=namespace,
    plugin = "qrb_ros::nn_inference::QrbRosInferenceNode",
    name = "nn_inference_node_encode",
    parameters = [
      {
        "backend_option": "/usr/lib/libQnnHtp.so",
        #"backend_option": "",
        "model_path": encode_model_path
      }],
    remappings = [
        ("qrb_inference_input_tensor", "/encode_qrb_inference_input_tensor"),
        ("qrb_inference_output_tensor", "/encode_qrb_inference_output_tensor"),
    ]
    )
    
    whisper_postprocess_node = Node(
        package='sample_speech_recognition',
        executable='qrb_ros_whisper_postprocess',
        namespace=namespace,
        output='screen',
    )
    
    nn_decode_inference_node = ComposableNode(
    package = "qrb_ros_nn_inference",
    namespace=namespace,
    plugin = "qrb_ros::nn_inference::QrbRosInferenceNode",
    name = "nn_inference_node_decode",
    parameters = [
      {
        "backend_option": "/usr/lib/libQnnHtp.so",
        #"backend_option": "",
        "model_path": decode_model_path
      }],
    remappings = [
        ("qrb_inference_input_tensor", "/decode_qrb_inference_input_tensor"),
        ("qrb_inference_output_tensor", "/decode_qrb_inference_output_tensor"),
    ]
    )
    
    # postprocess_node = Node(
        # package='sample_resnet101',
        # executable='qrb_ros_resnet101_posprocess',
        # namespace=namespace,
        # output='screen',
	# )
    
    container = ComposableNodeContainer(
        name = "container",
        namespace=namespace,
        package = "rclcpp_components",
        executable='component_container',
        output = "screen",
        composable_node_descriptions = [nn_encode_inference_node,nn_decode_inference_node]
    )
    
    return launch.LaunchDescription(declared_args + [whisper_preprocess_node,whisper_postprocess_node, container])
