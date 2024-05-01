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
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
import rospy
from frankapy import FrankaArm
from threading import Event
from pathlib import Path


repo_root_path = Path(__file__).parents[1]

def choose_robot_motion():
    robot_motions_path = repo_root_path / "frankaRobot" / "robotMotionPoints"
    robot_motions = dict([(str(i), p) for i, p in enumerate(robot_motions_path.iterdir())
                          if p.is_file and p.suffix == ".csv"])
    lines = [f'{key} {value.name}' for key, value in robot_motions.items()]
    print("Robot Motions:")
    print('\n'.join(lines) + '\n')
    robot_motion_key = None
    while robot_motion_key not in robot_motions:
        robot_motion_key = input(
            "Which robot motion should be used? (choose by index): ")
    return robot_motions[robot_motion_key]


def move_robot(fa:FrankaArm, event: Event, motion_csv):

	joints = pd.read_csv(motion_csv)

	# preprocessing
	joints = joints.iloc[:, 1:8]
	joints.iloc[:,6] -= np.deg2rad(45) 
	print(joints.head(5), '\n\n')
	fa.goto_joints(np.array(joints.iloc[0]),ignore_virtual_walls=True)
	fa.goto_gripper(0.02)
	
	while True:	
		try:	
			for i in range(joints.shape[0]):
				fa.goto_joints(np.array(joints.iloc[i]),ignore_virtual_walls=True,duration=1.5)
				#time.sleep(0.01)
			rospy.loginfo("move")

		except Exception as e:
			print(e)
			event.set()
			break
	
	print('fininshed .... !')



if __name__ == "__main__":
	global publish_output, big_time_digits

	motion_csv = choose_robot_motion()

	event = Event()
	
	# create robot controller instance
	fa = FrankaArm()
	scale = 1000000
	big_time_digits = int(rospy.get_time()/scale)*scale
	
	# subscribe robot data topic for contact detection module
	move_robot(fa, event, motion_csv)
	