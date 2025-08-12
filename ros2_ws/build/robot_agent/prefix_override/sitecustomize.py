import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/victor/TCC/robot-agent/ros2_ws/install/robot_agent'
