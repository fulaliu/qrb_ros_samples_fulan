# MIT License

# Copyright (c) 2022 OpenAI

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import numpy as np
import subprocess
import struct
import argparse
from typing import List, Tuple
from scipy import special as scipy_special  # type: ignore
from threading import Timer
#tokenizer 
import base64 
import tiktoken
import os
import decode
import audio_process
import shutil
import time
#for ros pkgs info
from ament_index_python.packages import get_package_share_directory
from qrb_ros_tensor_list_msgs.msg import Tensor, TensorList

TOKEN_SOT = 50257  # Start of transcript
TOKEN_EOT = 50256  # end of transcript
TOKEN_BLANK = 220  # " "
TOKEN_NO_TIMESTAMP = 50362
TOKEN_TIMESTAMP_BEGIN = 50363
TOKEN_NO_SPEECH = 50361

# Above this prob we deem there's no speech in the audio
NO_SPEECH_THR = 0.6

SAMPLE_BEGIN = 1  # first token is TOKEN_SOT

# https://github.com/openai/whisper/blob/v20230314/whisper/decoding.py#L600
NON_SPEECH_TOKENS = [
    1,
    2,
    7,
    8,
    9,
    10,
    14,
    25,
    26,
    27,
    28,
    29,
    31,
    58,
    59,
    60,
    61,
    62,
    63,
    90,
    91,
    92,
    93,
    357,
    366,
    438,
    532,
    685,
    705,
    796,
    930,
    1058,
    1220,
    1267,
    1279,
    1303,
    1343,
    1377,
    1391,
    1635,
    1782,
    1875,
    2162,
    2361,
    2488,
    3467,
    4008,
    4211,
    4600,
    4808,
    5299,
    5855,
    6329,
    7203,
    9609,
    9959,
    10563,
    10786,
    11420,
    11709,
    11907,
    13163,
    13697,
    13700,
    14808,
    15306,
    16410,
    16791,
    17992,
    19203,
    19510,
    20724,
    22305,
    22935,
    27007,
    30109,
    30420,
    33409,
    34949,
    40283,
    40493,
    40549,
    47282,
    49146,
    50257,
    50357,
    50358,
    50359,
    50360,
    50361,
]

class whisper_postprocess_node(Node):
    def __init__(self):
        super().__init__('qrb_ros_whisper_postprocess')
        self.publisher_result = self.create_publisher(String, 'whisper_result', 100)
        
        self.subscriber = self.create_subscription(
            TensorList,
            'encode_qrb_inference_output_tensor', 
            self.encode_callback,
            10
        )
        self.publisher = self.create_publisher(
            TensorList,
            'decode_qrb_inference_input_tensor',  # publish topic
            10
        )
        self.subscriber = self.create_subscription(
            TensorList,
            'decode_qrb_inference_output_tensor', 
            self.decode_callback,
            10
        )      
        self.k_cache_cross = np.zeros((4,6,64,1500), dtype=np.float32)
        self.v_cache_cross = np.zeros((4,6,1500,64), dtype=np.float32)
        #self.k_cache_self = np.zeros((4,6,64,244), dtype=np.float32)
        #self.v_cache_self = np.zeros((4,6,224,64), dtype=np.float32)
        self.k_cache_self = np.zeros((4,6,384,244), dtype=np.float32)
        self.v_cache_self = np.zeros((4,6,384,64), dtype=np.float32)
        self.decode_x = 0
        #np.array([[TOKEN_SOT]])
        self.decode_index = 0 
        self.logits = np.zeros((1,1,51864), dtype=np.float32)
        self.decode_time = 0
        self.decoded_tokens = [TOKEN_SOT]
        self.decode_exit_flag = False
        self.tokenizer = decode.get_tokenizer("/opt/model/gpt2.tiktoken")
        
        # sample_len = 224  # mean # of tokens to sample
        # num_decoder_blocks = 4
        # num_decoder_heads = 6
        # attention_dim = 384
        # self.k_cache_self = np.zeros(
            # (
                # num_decoder_blocks,
                # num_decoder_heads,
                # attention_dim // num_decoder_heads,
                # sample_len,
            # )
        # ).astype(np.float32)
        # self.v_cache_self = np.zeros(
            # (
                # num_decoder_blocks,
                # num_decoder_heads,
                # sample_len,
                # attention_dim // num_decoder_heads,
            # )
        # ).astype(np.float32)

    def send_to_decode_model(self):
        msg = TensorList()
        
        decode_x = np.array(self.decode_x, dtype=np.uint32).tobytes()
        #decode_x = self.decode_x.tobytes()
        #tensor list to input
        tensor = Tensor()
        tensor.data_type = 4
        tensor.name = "x"
        tensor.shape = [1,1]
        tensor.data = decode_x        
        msg.tensor_list.append(tensor)
        
        decode_index = np.array(self.decode_index, dtype=np.uint32).tobytes()
        #tensor list to input
        tensor = Tensor()
        tensor.data_type = 4
        tensor.name = "index"
        tensor.shape = [1,1]
        tensor.data = decode_index        
        msg.tensor_list.append(tensor)
        
        k_cache_cross = np.array(self.k_cache_cross, dtype=np.float32).tobytes()
        #tensor list to input
        tensor = Tensor()
        tensor.data_type = 2
        tensor.name = "k_cache_cross"
        tensor.shape = [4,6,64,1500]
        tensor.data = k_cache_cross        
        msg.tensor_list.append(tensor)
        
        v_cache_cross = np.array(self.v_cache_cross, dtype=np.float32).tobytes()
        #tensor list to input
        tensor = Tensor()
        tensor.data_type = 2
        tensor.name = "v_cache_cross"
        tensor.shape = [4,6,1500,64]
        tensor.data = v_cache_cross        
        msg.tensor_list.append(tensor)
        
        k_cache_self = np.array(self.k_cache_self, dtype=np.float32).tobytes()
        #tensor list to input
        tensor = Tensor()
        tensor.data_type = 2
        tensor.name = "k_cache_self"
        tensor.shape = [4,6,64,244]
        tensor.data = k_cache_self        
        msg.tensor_list.append(tensor)
        
        v_cache_self = np.array(self.v_cache_self, dtype=np.float32).tobytes()
        #tensor list to input
        tensor = Tensor()
        tensor.data_type = 2
        tensor.name = "v_cache_self"
        tensor.shape = [4,6,224,64]
        tensor.data = v_cache_self        
        msg.tensor_list.append(tensor)
        
        if self.decode_exit_flag == True :       
            try:
                text = self.tokenizer.decode(self.decoded_tokens[1:])  # remove TOKEN_SOT
            except Exception as e:
                text = "No sound detected" 
            self.publisher_result.publish(String(data=text.strip()))
        else:                
            self.publisher.publish(msg) 
            
    def encode_callback(self, encode_msg):
        try:
            for result_tensor in encode_msg.tensor_list:  # search tensor_list
                #self.get_logger().info(f'result_tensor shape is {result_tensor.shape}')
                if result_tensor.name == "k_cache":
                    self.k_cache_cross = result_tensor.data
                if result_tensor.name == "v_cache":
                    self.v_cache_cross = result_tensor.data
            self.decode_x = TOKEN_SOT  
            self.decode_index = 0
            self.send_to_decode_model()          
                        
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
                
        
    def decode_callback(self, decode_msg):
        try:
            for result_tensor in decode_msg.tensor_list:  # search tensor_list
                #self.get_logger().info(f'result_tensor shape is {result_tensor.shape}')
                print(f"result_tensor.name is {result_tensor.name}")
                if result_tensor.name == "k_cache":
                    self.k_cache_self = result_tensor.data
                if result_tensor.name == "v_cache":
                    self.v_cache_self = result_tensor.data
                if result_tensor.name == "logits":
                    self.logits = np.array(result_tensor.data).view(np.float32)
            if self.decode_exit_flag == False :
                next_token = np.argmax(self.logits)
                print(f"next_token is {next_token},self.decoded_tokens is {self.decoded_tokens}")
                
                # Filters
                # SuppressBlank
                if self.decode_time == 0:
                    self.logits[[TOKEN_EOT, TOKEN_BLANK]] = -np.inf
                # SuppressTokens
                self.logits[NON_SPEECH_TOKENS] = -np.inf

                # self.logits, logprobs = decode.apply_timestamp_rules(self.logits, self.decoded_tokens)
                # print(f"logprobs is {logprobs}")
                
                # if self.decode_time == 0:
                    # # detect no_speech
                    # no_speech_prob = np.exp(logprobs[TOKEN_NO_SPEECH])
                    # if no_speech_prob > NO_SPEECH_THR:
                        # self.decode_exit_flag = True
                    # print(f"in the decode time 0, no_speech_prob is {no_speech_prob}, NO_SPEECH_THR is {NO_SPEECH_THR}")
                    
                # temperature = 0
                next_token = np.argmax(self.logits)
                if next_token == TOKEN_EOT:
                    self.decode_exit_flag = True

                print(f"NO_SPEECH_THR is {NO_SPEECH_THR}, next_token is {next_token},TOKEN_EOT is {TOKEN_EOT}")
                #sum_logprobs += logprobs[next_token]
                self.decode_x = next_token
                print("Before increment:", self.decode_index)
                self.decode_index += 1
                print("After increment:", self.decode_index)
                self.decode_time += 1
                self.decoded_tokens.append(int(next_token))
                
                self.send_to_decode_model()
            else:
                print(f"decode exit flag is True, only output whisper output")
                        
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

node = None
def get_decode_exit_status():
    global node
    if node is not None and node.decode_exit_flag == True:
        return True
    else:
        return False
        
def set_decode_exit_status():
    global node
    if node is not None:
        node.decode_exit_flag = False
        node.decode_x = TOKEN_SOT  
        node.decode_index = 0


def main(args=None):
    
    global node    
    rclpy.init(args=args)
    node = whisper_postprocess_node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    

if __name__ == '__main__':   
    main()
    
     
