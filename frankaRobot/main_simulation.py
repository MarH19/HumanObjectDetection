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
		conda activate frankapyenv
		source /opt/ros/noetic/setup.bash
		source /home/mindlab/franka/franka-interface/catkin_ws/devel/setup.bash --extend
		source /home/mindlab/franka/frankapy/catkin_ws/devel/setup.bash --extend
		/home/mindlab/miniconda3/envs/frankapyenv/bin/python3 /home/mindlab/humanObjectDetection/frankaRobot/main_simulation.py


5. run save data node
	-open another terminal
		source /opt/ros/noetic/setup.bash
		source /home/mindlab/franka/franka-interface/catkin_ws/devel/setup.bash --extend
		source /home/mindlab/franka/frankapy/catkin_ws/devel/setup.bash --extend
		/home/mindlab/miniconda3/envs/frankapyenv/bin/python3 /home/mindlab/humanObjectDetection/frankaRobot/saveDataNode.py

# to chage publish rate of frankastate go to :
sudo nano /home/mindlab/franka/franka-interface/catkin_ws/src/franka_ros_interface/launch/franka_ros_interface.launch
"""
"""
target_position = ['e0','e1','e2','e3','e4','e5','e6']
target_velocity = ['de0','de1','de2','de3','de4','de5','de6']
target_torques = ['etau_J0','etau_J1', 'etau_J2', 'etau_J3', 'etau_J4', 'etau_J5', 'etau_J6']

"""



import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import json
from pathlib import Path
from threading import Event
from typing import Type

import numpy as np
import pandas as pd
import rospy
import torch
from franka_interface_msgs.msg import RobotState
from frankapy import FrankaArm
from importModel import import_lstm_models, import_lstm_models_old
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from std_msgs.msg import Float64
from torchvision import transforms
import torch.nn as nn
from rnn_models import LSTMModel

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


repo_root_path = Path(__file__).parents[1]

########### Define parameters for contact detection NICO,LARS,MAXI
labels_contact_binary = {0: "no contact", 1:  "contact"}
binary_contact_window_lenght = 5
binary_contact_input_size = 21


# Set device for PyTorch models and select first GPU cuda:0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if device.type == "cuda":
    torch.cuda.get_device_name()

#Set path to model    
contact_binary_path = repo_root_path / \
    'AIModels' / 'trainedModels' / 'simulation' / 'real_lstm-binary-prediction.pth'

robot_motion_path = choose_robot_motion()

######### Model load for contact detection NICO,LARS,MAXI
model_contact_binary = LSTMModel(
    input_size=binary_contact_input_size, hidden_size=126,
    num_layers=2, output_size=1)
model_contact_binary.load_state_dict(
    torch.load(str(contact_binary_path.absolute()), map_location='cpu'))
model_contact_binary.eval()
print("Lars und Maxi eues Model hender ääääntli ufem Roboter! Lets goooo!")

# Move PyTorch models to the selected device
model_contact_binary = model_contact_binary.to(device)



window_contact_binary= np.zeros(
    [binary_contact_window_lenght, binary_contact_input_size])





def contact_detection(data):
    global window_contact_binary
    start_time = rospy.get_time()
    position = np.array(data.q)
    velocity = np.array(data.dq)
    torque = np.array(data.tau_J)
    features = np.concatenate([position, velocity, torque])
    features = features.reshape((1, binary_contact_input_size)) ####Ist das richtig?
    window_contact_binary = np.append(
        window_contact_binary[1:, :], features, axis=0)
    features_tensor_contact = torch.tensor(window_contact_binary, dtype=torch.float32).unsqueeze(
        0).to(device)  # gives (1,window_size,feature_number)

    with torch.no_grad():
        model_out_contact = model_contact_binary(features_tensor_contact)
        contact_binary_prediction = model_contact_binary.get_predictions(model_out_contact)
        detection_duration = rospy.get_time() - start_time
        rospy.loginfo(f'Contact binary prediction: {contact_binary_prediction}')
    




def move_robot(fa: FrankaArm, event: Event):

    joints = pd.read_csv(str(robot_motion_path.absolute()))

    # preprocessing
    joints = joints.iloc[:, 1:8]
    joints.iloc[:, 6] -= np.deg2rad(45)
    print(joints.head(5), '\n\n')
    fa.goto_joints(np.array(joints.iloc[0]), ignore_virtual_walls=True)
    fa.goto_gripper(0.02)

    while True:
        try:
            for i in range(joints.shape[0]):
                fa.goto_joints(
                    np.array(joints.iloc[i]), ignore_virtual_walls=True, duration=1.75)
                # time.sleep(0.01)

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
    big_time_digits = int(rospy.get_time() / scale) * scale
    # subscribe robot data topic for contact detection module
    rospy.Subscriber(name="/robot_state_publisher_node_1/robot_state", data_class=RobotState,
                     callback=contact_detection)  # , callback_args=update_state)#,queue_size = 1)
    move_robot(fa, event)