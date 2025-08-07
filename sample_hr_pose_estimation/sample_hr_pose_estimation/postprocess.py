import numpy as np
import cv2
import os
import math
from collections import deque
from ament_index_python.packages import get_package_share_directory
from launch.logging import get_logger

package_share_directory = get_package_share_directory("sample_hr_pose_estimation")

# 👇 全局定义一次即可
history_buffer = deque(maxlen=5)

import sys
import logging

logging.basicConfig(
    filename='output.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def postprocess(image,output_tensor):
    height, width, _ = image.shape
    output_data = output_tensor.reshape((64, 48, 17))
    
    max_indices = []
    max_values = []
    for z in range(output_data.shape[2]):
        max_index = np.unravel_index(np.argmax(output_data[:,:,z]), output_data[:,:,z].shape)
        max_value = np.max(output_data[:, :, z])
        max_values.append(max_value)
        max_indices.append(max_index)

    greater_than_indices = []
    
    for z in range(output_data.shape[2]):
        indices = np.where(output_data[:, :, z] > 0.05)
        greater_than_indices.append(indices)


    skeleton = [
        [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
        [5, 11], [6, 12], [5, 6],
        [5, 7], [6, 8],
        [7, 9], [8, 10],
        [1, 2],
        [0, 1], [0, 2],
        [1, 3], [2, 4],
        [3, 5], [4, 6]
    ]
    
    
    
             
    for z, max_index in enumerate(max_indices):
        x, y = max_index
        center_x = int(x / 64 * width)
        center_y = int(y / 48 * height)
        # 获取关键点
        if z == 1:
            eye_left = (center_x,center_y)
        if z == 2:
            eye_right = (center_x,center_y)
        if z == 3:
            ear_left = (center_x,center_y)
        if z == 4:
            ear_right = (center_x,center_y)
        if z == 5:
            shoulder_left = (center_x,center_y)
        if z == 6:
            shoulder_right = (center_x,center_y)
        #print(f"point {z},  is {center_x},{center_y}")
        if max_values[z] > 0.1:
            cv2.circle(image, (center_x, center_y), 12, (0, 0, 255), -1)

    # 计算角度
    #neck_angle = calculate_angle(ear_x,ear_y, shoulder_x,shoulder_y,hip_x,hip_y)
    #back_angle = calculate_angle(shoulder_x,shoulder_y, hip_x,hip_y,knee_x,knee_y)

    # 判断是否异常
    #if neck_angle < 50:
    # if back_angle < 150:
        # cv2.putText(image, "!!! Abnormality of back flexion", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
        
    result = is_neck_forward_tilted_v2(eye_left, eye_right,ear_left, ear_right,shoulder_left, shoulder_right,age_group='adult')
    if result["neck_forward_now"]:
        cv2.putText(image, "!!! Now Poor forward posture of cervical spine ", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    if result["neck_forward_consistent"]:
        cv2.putText(image, "!!! consistent Poor forward posture of cervical spine ", (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)    
    

    logging.info(f"是否当前前倾： {result['neck_forward_now']}")
    logging.info(f"是否持续前倾： {result['neck_forward_consistent']}")
    logging.info(f"眼睛-肩膀角度： {result['eye_shoulder_angle']}")
    logging.info(f"耳朵-肩膀角度： {result['ear_shoulder_angle']}")
        

    


        
    for connection in skeleton:
        start_point = max_indices[connection[0]]
        end_point = max_indices[connection[1]]
        start_x, start_y = int(start_point[0] / 64 * width), int(start_point[1] / 48 * height)
        end_x, end_y = int(end_point[0] / 64 * width), int(end_point[1] / 48 * height)
        if max_values[connection[0]] > 0.1 and max_values[connection[1]] > 0.1:
            cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 0, 0), 8)

    return image


# 计算角度函数
def calculate_angle(top_x,top_y, mid_x,mid_y,down_x,down_y):
    #print(f" the point location is {top_x},{top_y}, {mid_x},{mid_y},{down_x},{down_y}")
    a_x = top_x
    a_y = top_y
    b_x = mid_x
    b_y = mid_y
    c_x = down_x
    c_y = down_y
    ab = [a_x - b_x, a_y - b_y]
    cb = [c_x - b_x, c_y - b_y]
    dot = ab[0]*cb[0] + ab[1]*cb[1]
    ab_len = math.sqrt(ab[0]**2 + ab[1]**2)
    cb_len = math.sqrt(cb[0]**2 + cb[1]**2)
    angle = math.acos(dot / (ab_len * cb_len))
    return math.degrees(angle)

def is_neck_forward_tilted_v2(
    eye_left, eye_right,
    ear_left, ear_right,
    shoulder_left, shoulder_right,
    age_group='adult',
):
    """
    判断是否存在脖子前倾，考虑多点、年龄群差异与时间维度。

    参数:
        eye_left, eye_right: 左右眼 (x, y)
        ear_left, ear_right: 左右耳 (x, y)
        shoulder_left, shoulder_right: 左右肩 (x, y)
        age_group (str): 'child' 或 'adult'，用于调节角度阈值
        history_buffer (deque): 用于存储前几帧结果，判定是否持续异常

    返回:
        dict: 包含是否异常、角度详情与时间逻辑分析
    """
    global history_buffer 
    def midpoint(p1, p2):
        return ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)

    def angle_with_vertical(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle_rad = math.atan2(dx, dy)
        angle_deg = abs(math.degrees(angle_rad))
        return angle_deg

    # 阈值根据年龄群调整
    threshold = 10 if age_group == 'child' else 15

    # 计算关键点中点
    eye_center = midpoint(eye_left, eye_right)
    ear_center = midpoint(ear_left, ear_right)
    shoulder_center = midpoint(shoulder_left, shoulder_right)

    # 计算角度
    eye_shoulder_angle = angle_with_vertical(eye_center, shoulder_center)
    ear_shoulder_angle = angle_with_vertical(ear_center, shoulder_center)

    # 当前帧是否异常
    current_status = (eye_shoulder_angle > 20) and (ear_shoulder_angle > 8)

    # 加入时间判断：连续超过 N 次才认为是持续异常
    history_buffer.append(current_status)
    consistent_abnormal = history_buffer.count(True) >= history_buffer.maxlen

    return {
        "neck_forward_now": current_status,
        "neck_forward_consistent": consistent_abnormal,
        "eye_shoulder_angle": round(eye_shoulder_angle, 2),
        "ear_shoulder_angle": round(ear_shoulder_angle, 2),
        "age_group": age_group,
        "threshold_angle": threshold
    }