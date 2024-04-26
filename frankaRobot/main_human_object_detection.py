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
		/home/mindlab/miniconda3/envs/frankapyenv/bin/python3 /home/mindlab/humanObjectDetection/frankaRobot/main_human_object_detection.py


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
from importModel import import_lstm_models
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from std_msgs.msg import Float64
from torchvision import transforms

from ModelGeneration.model_generation import choose_model_class
from ModelGeneration.rnn_models import RNNModel

repo_root_path = Path(__file__).parents[1]


def choose_trained_model(model_class: Type[RNNModel]):
    trained_models_path = repo_root_path / "ModelGeneration" / "TrainedModels"
    with open(str((trained_models_path / "RnnModelsParameters.json").absolute()), 'r') as f:
        model_params_list = json.load(f)
    model_params_list = [
        m for m in model_params_list if m["model_name"].startswith(model_class.__name__ + "_")]
    model_params_list = sorted(
        model_params_list, key=lambda d: d['model_name'])

    lines = [f'{i} {v["model_name"]}' for i, v in enumerate(model_params_list)]
    print("Trained model files:")
    print('\n'.join(lines) + '\n')
    model_params_index = None
    while model_params_index not in np.arange(0, len(model_params_list), 1):
        model_params_index = int(input(
            "Which trained model parameters should be used? (choose by index): "))
    return model_params_list[model_params_index]


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


# choose model type and trained model parameters
classification_model_class = choose_model_class()
classification_model_params = choose_trained_model(classification_model_class)
robot_motion_path = choose_robot_motion()
print()

# Define parameters for the contact detection / localization models
num_features_lstm = 4
window_length = 28
features_num = 28  # 4 variables are and 7 joints -> 4*7 = 28
dof = 7

# Define parameters for the classification (human object detection) model
labels_classification = {0: "hard", 1: "plasticbottle", 2: "soft"}
window_classification_length = 40
classification_model_input_size = 14

# Set device for PyTorch models and select first GPU cuda:0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if device.type == "cuda":
    torch.cuda.get_device_name()

# load models
contact_detection_path = repo_root_path / \
    'AIModels' / 'trainedModels' / 'contactDetection' / \
    'trainedModel_01_24_2024_11_18_01.pth'

model_name = classification_model_params["model_name"]
classification_path = repo_root_path / "ModelGeneration" / \
    "TrainedModels" / f"{model_name}.pth"

model_contact, labels_map_contact = import_lstm_models(
    PATH=str(contact_detection_path.absolute()), num_features_lstm=num_features_lstm)

model_classification = classification_model_class(
    input_size=classification_model_input_size, hidden_size=classification_model_params[
        "hyperparameters"]["hidden_size"],
    num_layers=classification_model_params["hyperparameters"]["num_layers"], output_size=3)
model_classification.load_state_dict(
    torch.load(str(classification_path.absolute()), map_location='cpu'))
model_classification.eval()
print("contact classification (human object detection) model is loaded!")

# Move PyTorch models to the selected device
model_contact = model_contact.to(device)
model_classification = model_classification.to(device)

# Define transformation for input data
transform = transforms.Compose([transforms.ToTensor()])
window = np.zeros([window_length, features_num])
window_classification = np.zeros(
    [window_classification_length, classification_model_input_size])
# Create message for publishing model output (will be used in saceDataNode.py)
model_msg = Floats()


def contact_detection(data):
    global window, publish_output, big_time_digits
    global window_classification
    start_time = rospy.get_time()
    e_q = np.array(data.q_d) - np.array(data.q)
    e_dq = np.array(data.dq_d) - np.array(data.dq)
    tau_J = np.array(data.tau_J)
    tau_ext = np.array(data.tau_ext_hat_filtered)
    tau_ext = np.multiply(tau_ext, 0.5)

    new_row = np.hstack((tau_J, tau_ext, e_q, e_dq))
    new_row = new_row.reshape((1, features_num))

    window = np.append(window[1:, :], new_row, axis=0)

    # change the order for lstm
    lstmDataWindow = []
    for j in range(dof):
        # tau(t), tau_ext(t), e(t), de(t)

        if num_features_lstm == 4:
            column_index = [j, j + dof, j + dof * 2, j + dof * 3]
        elif num_features_lstm == 2:
            column_index = [j + dof * 2, j + dof * 3]
        elif num_features_lstm == 3:
            column_index = [j + dof, j + dof * 2, j + dof * 3]

        join_data_matix = window[:, column_index]
        lstmDataWindow.append(join_data_matix.reshape(
            (1, num_features_lstm * window_length)))

    lstmDataWindow = np.vstack(lstmDataWindow)

    # data input prep for classification
    features = np.array(data.tau_J_d) - np.array(data.tau_J)  # etau
    features = np.concatenate([features, e_q])
    features = features.reshape((1, classification_model_input_size))
    window_classification = np.append(
        window_classification[1:, :], features, axis=0)

    features_tensor = torch.tensor(window_classification, dtype=torch.float32).unsqueeze(
        0).to(device)  # gives (1,window_size,feature_number)

    with torch.no_grad():
        # mit torch.tensor(X_test, dtype=torch.float32) ausprobieren
        data_input = transform(lstmDataWindow).to(device).float()
        model_out = model_contact(data_input)
        model_out = model_out.detach()
        output = torch.argmax(model_out, dim=1)

    contact = output.cpu().numpy()[0]
    if contact == 1:
        with torch.no_grad():
            model_out = model_classification(features_tensor)
            contact_object_prediction = model_classification.get_predictions(
                model_out)

        detection_duration = rospy.get_time() - start_time
        rospy.loginfo(
            f'detection duration: {detection_duration}, classification prediction: {labels_classification[int(contact_object_prediction)]}')

    else:
        detection_duration = rospy.get_time() - start_time
        contact_object_prediction = -1
        # rospy.loginfo('detection duration: %f, there is no contact',detection_duration)
        # publish_output.publish([detection_duration, contact, contact, contact])
    start_time = np.array(start_time).tolist()
    time_sec = int(start_time)
    time_nsec = start_time - time_sec
    model_msg.data = np.array([time_sec - big_time_digits, time_nsec,
                              detection_duration, contact, contact_object_prediction], dtype=np.complex128)
    model_pub.publish(model_msg)


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
    model_pub = rospy.Publisher(
        "/model_output", numpy_msg(Floats), queue_size=1)
    move_robot(fa, event)
