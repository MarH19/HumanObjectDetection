#!/usr/bin/env python3
"""
By Maryam Rezayati

# How to run?
1. unlock robot
	-turn on the robot (wait until it has a solid yellow)
	-connect to the robot desk with the ID (172.16.0.2 or 192.168.15.33)
	-unlock the robot
	-the robot light should be blue
	-unlock the robot and activate FCI

2. run frankapy
	-open an terminal
		conda activate frankapyenv
		bash /home/mindlab/franka/frankapy/bash_scripts/start_control_pc.sh -i localhost

3. run sensor node
	-open another temrinal
		source /opt/ros/noetic/setup.bash
		/home/mindlab/miniconda3/envs/frankapyenv/bin/python /home/mindlab/humanObjectDetection/dataLabeling/sensorNode.py

4. run robot node
	-open another terminal 
		conda activate frankapyenv
		source /opt/ros/noetic/setup.bash
		source /home/mindlab/franka/franka-interface/catkin_ws/devel/setup.bash --extend
		source /home/mindlab/franka/frankapy/catkin_ws/devel/setup.bash --extend
	
		/home/mindlab/miniconda3/envs/frankapyenv/bin/python3 /home/mindlab/humanObjectDetection/frankaRobot/moveRobot.py

5. run save data node
	-open another terminal
		source /opt/ros/noetic/setup.bash
		source /home/mindlab/franka/franka-interface/catkin_ws/devel/setup.bash --extend
		source /home/mindlab/franka/frankapy/catkin_ws/devel/setup.bash --extend
	
		/home/mindlab/miniconda3/envs/frankapyenv/bin/python3 /home/mindlab/humanObjectDetection/frankaRobot/saveDataNode.py

# to chage publish rate of frankastate go to : 
sudo nano /home/mindlab/franka/franka-interface/catkin_ws/src/franka_ros_interface/launch/franka_ros_interface.launch
"""

## import required libraries 
import os
import numpy as np
import pandas as pd
import time

import torch
from torchvision import transforms

import rospy
from std_msgs.msg import Float64
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from frankapy import FrankaArm
from franka_interface_msgs.msg import RobotState
from threading import Thread
from threading import Event
from importModel import import_lstm_models

# Set the main path
main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'

# Define paths for joint motion data
joints_data_path = main_path + 'frankaRobot/robotMotionPoints/robotMotionJointData_02_22_2024_11:01:21.csv'

def move_robot(fa:FrankaArm, event: Event):

	joints = pd.read_csv(joints_data_path)

	# preprocessing
	joints = joints.iloc[:, 1:8]
	joints.iloc[:,6] -= np.deg2rad(45) 
	print(joints.head(5), '\n\n')
	fa.goto_joints(np.array(joints.iloc[0]),ignore_virtual_walls=True)
	fa.goto_gripper(0.02)
	
	while True:	
		try:	
			for i in range(joints.shape[0]):
				fa.goto_joints(np.array(joints.iloc[i]),ignore_virtual_walls=True,duration=1.25)
				#time.sleep(0.01)
			rospy.loginfo("move")

		except Exception as e:
			print(e)
			event.set()
			break
	
	print('fininshed .... !')



if __name__ == "__main__":
	global publish_output, big_time_digits
	event = Event()
	# create robot controller instance
	fa = FrankaArm()
	scale = 1000000
	big_time_digits = int(rospy.get_time()/scale)*scale
	# subscribe robot data topic for contact detection module
	move_robot(fa, event)
	
