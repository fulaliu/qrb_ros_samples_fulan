import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/ubuntu/ASR/qrb_ros_samples_fulan/ai_audio/sample_speech_recognition/install/sample_speech_recognition'
