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
from torchvision import transforms

from _util.util import choose_robot_motion, user_input_choose_from_list
from ModelGeneration.model_generation import choose_rnn_model_class
from ModelGeneration.rnn_models import RNNModel
from ModelGeneration.transformer.model import ConvTran

repo_root_path = Path(__file__).parents[1]
classification_model_input_size = 21
window_classification_length = 40
labels_classification = {0: "hard", 1: "pvc_tube", 2: "soft"}


def choose_model_type():
    return user_input_choose_from_list(["RNN", "Transformer"], "Model Types")


def choose_trained_rnn_model(model_class: Type[RNNModel]):
    trained_models_path = repo_root_path / "ModelGeneration" / "TrainedModels"
    with open(str((trained_models_path / "RnnModelsParameters.json").absolute()), 'r') as f:
        model_params = json.load(f)
    model_params = [
        m for m in model_params if m["model_name"].startswith(model_class.__name__ + "_")]
    model_params = sorted(
        model_params, key=lambda d: d['model_name'])
    return user_input_choose_from_list(model_params, "Trained model files", lambda v: v["model_name"])


def choose_trained_transformer_model():
    transformer_results_path = repo_root_path / \
        "ModelGeneration" / "transformer" / "Results"
    trained_model_paths = []
    for sp in transformer_results_path.iterdir():
        if sp.is_dir() and sp.name != "old_models":
            trained_model_paths += [p for _,
                                    p in enumerate(sp.iterdir()) if p.is_dir()]
    return user_input_choose_from_list(trained_model_paths, "Trained Transformer models", lambda p: f"{p.parent.name}/{p.name}")


def load_rnn_classification_model(model_class: type[RNNModel], params):
    model_name = rnn_model_params["model_name"]
    classification_path = repo_root_path / "ModelGeneration" / \
        "TrainedModels" / f"{model_name}.pth"
    model_classification = rnn_model_class(
        input_size=classification_model_input_size, hidden_size=rnn_model_params[
            "hyperparameters"]["hidden_size"],
        num_layers=rnn_model_params["hyperparameters"]["num_layers"], output_size=len(labels_classification))
    model_classification.load_state_dict(torch.load(
        str(classification_path.absolute()), map_location='cpu'))
    return model_classification


def load_transformer_classification_model(model_path: Path):
    checkpoint_path = model_path / "checkpoints" / "model_last.pth"
    config_path = model_path / "configuration.json"
    with open(str((config_path).absolute()), 'r') as f:
        config = json.load(f)
    config['Data_shape'] = [
        1, classification_model_input_size, window_classification_length]
    model_classification = ConvTran(
        config=config, num_classes=len(labels_classification))
    saved_params = torch.load(
        str(checkpoint_path.absolute()), map_location='cpu')
    model_classification.load_state_dict(saved_params["state_dict"])
    return model_classification, config


def normalize_window(window):
    params = rnn_model_params if model_type == "RNN" else transformer_config
    if "normalization_max" in params and "normalization_min" in params and len(params["normalization_max"]) == window.shape[1] and len(params["normalization_min"]) == window.shape[1]:
        for i, max in enumerate(params["normalization_max"]):
            min = params["normalization_min"][i]
            window[:, i] = (window[:, i] - min) / (max - min)
    elif "normalization_mean" in params and "normalization_std" in params and len(params["normalization_mean"]) == window.shape[1] and len(params["normalization_std"]) == window.shape[1]:
        for i, mean in enumerate(params["normalization_mean"]):
            std = params["normalization_std"][i]
            window[:, i] = (window[:, i] - mean) / std
    return window


# choose trained model
model_type = choose_model_type()
model_classification, rnn_model_params, transformer_config = None, None, None
if model_type == "RNN":
    rnn_model_class = choose_rnn_model_class()
    rnn_model_params = choose_trained_rnn_model(rnn_model_class)
    model_classification = load_rnn_classification_model(
        rnn_model_class, rnn_model_params)
elif model_type == "Transformer":
    transformer_model_path = choose_trained_transformer_model()
    model_classification, transformer_config = load_transformer_classification_model(
        transformer_model_path)

robot_motion_path = choose_robot_motion()
print()

# Define parameters for the contact detection / localization models
contact_detection_num_features_lstm = 4
contact_detection_window_length = 28
contact_detection_nof_features = 28
contact_detection_dof = 7

# Set device for PyTorch models and select first GPU cuda:0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load contact detection model
contact_detection_path = repo_root_path / \
    'AIModels' / 'trainedModels' / 'contactDetection' / \
    'trainedModel_01_24_2024_11_18_01.pth'
model_contact, labels_map_contact = import_lstm_models(
    PATH=str(contact_detection_path.absolute()), num_features_lstm=contact_detection_num_features_lstm)

# Move PyTorch models to the selected device
model_contact = model_contact.to(device)
model_classification = model_classification.to(device)
model_classification.eval()

# Define transformation for contact detection model input data
contact_detection_transform = transforms.Compose([transforms.ToTensor()])

# Initialize remaining variables
contact_detection_window = np.zeros(
    [contact_detection_window_length, contact_detection_nof_features])
classification_window = np.zeros(
    [window_classification_length, classification_model_input_size])
model_output_msg = Floats()
classification_counter = 0


def contact_predictions(data):
    global contact_detection_window, publish_output, big_time_digits
    global classification_window, classification_counter

    start_time = rospy.get_time()

    e_q = np.array(data.q_d) - np.array(data.q)
    e_dq = np.array(data.dq_d) - np.array(data.dq)
    tau_J = np.array(data.tau_J)
    tau_ext = np.multiply(np.array(data.tau_ext_hat_filtered), 0.5)

    # Data for contact detection
    contact_detection_row = np.hstack((tau_J, tau_ext, e_q, e_dq))
    contact_detection_row = contact_detection_row.reshape(
        (1, contact_detection_nof_features))
    contact_detection_window = np.append(
        contact_detection_window[1:, :], contact_detection_row, axis=0)
    lstmDataWindow = []
    for j in range(contact_detection_dof):
        column_index = [j, j + contact_detection_dof, j +
                        contact_detection_dof * 2, j + contact_detection_dof * 3]
        join_data_matrix = contact_detection_window[:, column_index]
        lstmDataWindow.append(join_data_matrix.reshape(
            (1, contact_detection_num_features_lstm * contact_detection_window_length)))
    lstmDataWindow = np.vstack(lstmDataWindow)

    # Data for classification (human/object detection)
    classification_row = np.concatenate([tau_J, e_q, e_dq])
    classification_row = classification_row.reshape(
        (1, classification_model_input_size))
    classification_window = np.append(
        classification_window[1:, :], classification_row, axis=0)

    # Run contact detection model
    with torch.no_grad():
        data_input = contact_detection_transform(
            lstmDataWindow).to(device).float()
        model_out = model_contact(data_input)
        model_out = model_out.detach()
        output = torch.argmax(model_out, dim=1)

    # Run classification model
    contact = output.cpu().numpy()[0]
    contact_object_prediction = -1
    if contact == 1:
        # only do a classification every 3rd time a contact is detected (0, 1, 2)
        if classification_counter == 2:
            with torch.no_grad():
                # normalize data if normalization was done during model training
                classification_window = normalize_window(classification_window)
                if model_type == "RNN":
                    classification_tensor = torch.tensor(
                        classification_window, dtype=torch.float32).unsqueeze(0).to(device)
                    model_out = model_classification(classification_tensor)
                    contact_object_prediction = model_classification.get_predictions(
                        model_out)
                elif model_type == "Transformer":
                    classification_tensor = torch.tensor(
                        np.swapaxes(classification_window, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)
                    model_out = model_classification(classification_tensor)
                    contact_object_prediction = torch.argmax(
                        torch.nn.functional.softmax(model_out, dim=1), dim=1).cpu().numpy()
        # increment / reset classification counter
        classification_counter = (classification_counter + 1) % 3
    else:
        # reset classification counter if no contact -> classification model runs only
        classification_counter = 0

    all_models_prediction_duration = rospy.get_time() - start_time
    rospy.loginfo(
        f'all modelsprediction duration: {all_models_prediction_duration}, classification prediction: {labels_classification[int(contact_object_prediction)]}')

    start_time = np.array(start_time).tolist()
    time_sec = int(start_time)
    time_nsec = start_time - time_sec
    model_output_msg.data = np.array([time_sec - big_time_digits, time_nsec,
                                      all_models_prediction_duration, contact, contact_object_prediction], dtype=np.complex128)
    model_pub.publish(model_output_msg)


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
    rospy.Subscriber(name="/robot_state_publisher_node_1/robot_state",
                     data_class=RobotState, callback=contact_predictions)
    model_pub = rospy.Publisher(
        "/model_output", numpy_msg(Floats), queue_size=1)
    move_robot(fa, event)
