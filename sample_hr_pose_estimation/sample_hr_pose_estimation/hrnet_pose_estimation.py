import os
import numpy as np
import cv2
import rclpy
import preprocess
import postprocess
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from qrb_ros_tensor_list_msgs.msg import Tensor, TensorList
from ament_index_python.packages import get_package_share_directory

package_share_directory = get_package_share_directory("sample_hr_pose_estimation")

# raw image subscriber -> preprocess -> nn_inference_input_tensor publisher -> nn_inference_output_tensor subscriber -> postprocess -> pose_estimation_results publisher

class hrnet_pose_estimation(Node):
    def __init__(self):
        super().__init__('hrnet_pose_estimation_node')

        self.raw_image_subscription = self.create_subscription(Image, 'image_raw', self.raw_image_callback, 10)
        self.pose_detection_result_image_publisher = self.create_publisher(Image, 'pose_estimation_results', 10)
        self.nn_inference_output_tensor_subscription = self.create_subscription(TensorList, 'qrb_inference_output_tensor', self.nn_inference_callback, 10)
        self.nn_inference_input_tensor_publisher = self.create_publisher(TensorList, 'qrb_inference_input_tensor', 10)

        self.raw_image_processed_flag = True
        self.preprocessed_image = None

        self.bridge = CvBridge()

    # define call back function
    def raw_image_callback(self, msg):
        #self.get_logger().info('Received raw_image message')

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        preprocessed_image = preprocess.preprocess(cv_image)

        if self.raw_image_processed_flag == True:
            self.cv_image = cv_image
            nn_inference_input_tensor_msg = self.make_nn_inference_input_tensor_msg(msg, preprocessed_image)
            self.nn_inference_input_tensor_publisher.publish(nn_inference_input_tensor_msg)
            #self.get_logger().info("Publish nn_inference_input_tensor_msg")
            self.raw_image_processed_flag = False
        else:
            #self.get_logger().info("Skip nn_inference_input_tensor_msg")
           return

    def make_nn_inference_input_tensor_msg(self, original_msg, preprocessed_image):

        nn_inference_input_tensor_msg = TensorList()
        nn_inference_input_tensor_msg.header = original_msg.header

        tensor = Tensor()
        tensor.name = "pose detection nn_inference_input_tensor"
        tensor.data_type = 2    #float32
        tensor.shape = preprocessed_image.shape
        tensor.data = preprocessed_image.tobytes()

        nn_inference_input_tensor_msg.tensor_list.append(tensor)
        return nn_inference_input_tensor_msg

    def nn_inference_callback(self, nn_inference_output_tensor_msg):
        self.get_logger().info('Received nn_inference_output_tensor message')
        for result in nn_inference_output_tensor_msg.tensor_list:
            output_np_array = np.array(result.data)
            output_np_array = output_np_array.view(np.float32)
            result_image = postprocess.postprocess(self.cv_image,output_np_array)
            self.publish_result_image_message(result_image)

        self.raw_image_processed_flag = True

    def publish_result_image_message(self, image):
        ros_image = self.bridge.cv2_to_imgmsg(image,encoding="bgr8")
        self.pose_detection_result_image_publisher.publish(ros_image)
        self.get_logger().info('Publisher pose_estimation_results image message')


def main(args=None):
    rclpy.init(args=args)
    node = hrnet_pose_estimation()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
