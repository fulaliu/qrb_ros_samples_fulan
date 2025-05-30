# QRB ROS Samples

## Overview

This repository is a comprehensive collection of QRB ROS (Robot Operating System) example codes. It serves as a valuable resource for developers and enthusiasts looking to explore and implement QRB functionalities within the ROS framework. Each example is designed to demonstrate specific features and use cases, providing a practical guide to enhance your understanding and application of QRB in ROS environments.

## List of samples

|           | Samples                                                      | Description                                                  | Model Source                                                 |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| AI Audio  | [speech recognition](ai_audio/sample_speech_recognition/)    | captures the audio input and publishes the ros topic with the speech recognition result | [Whisper-Tiny-En - Qualcomm AI Hub](https://aihub.qualcomm.com/iot/models/whisper_tiny_en?domain=Audio) |
| AI Vision | [sample resnet101 quantized](ai_vision/sample_resnet101_quantized/) | Python-based classify images ROS node that uses QNN for model inference. | [ResNet101Quantized - Qualcomm AI Hub](https://aihub.qualcomm.com/iot/models/resnet101_quantized?searchTerm=resn) |

## System Requirements

- QIRP SDK . [qualcomm-linux/meta-qcom-robotics-sdk](https://github.com/qualcomm-linux/meta-qcom-robotics-sdk)
- ROS 2 Jazzy and later.

## Supported Platforms

This package is designed and tested to be compatible with ROS 2 Jazzy running on Qualcomm RB3 gen2.

| Hardware                        | Software                |
| ------------------------------- | ----------------------- |
| RB3 gen2                        | LE.QCROBOTICS.1.0       |

## Contributions

Thanks for your interest in contributing to qrb_ros_interfaces! Please read our [Contributions Page](CONTRIBUTING.md) for more information on contributing features or bug fixes. We look forward to your participation!

## License

qrb_ros_samples is licensed under the BSD 3-clause "New" or "Revised" License.

Check out the [LICENSE](LICENSE) for more details.
