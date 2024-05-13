import json
from pathlib import Path
from threading import Event
from typing import Type
import torch
import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from _util.util import choose_robot_motion, user_input_choose_from_list,choose_dataset, normalize_dataset,choose_normalization_mode
from ModelGeneration.rnn_models import RNNModel
from ModelGeneration.transformer.model import ConvTran
from ModelGeneration.model_generation import choose_rnn_model_class
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from dotenv import find_dotenv, load_dotenv
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
    # TODO nicht nur window sondern ganzes dataset
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

load_dotenv(find_dotenv())
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


sub_repo, X_file = choose_dataset()
#normalization_mode = choose_normalization_mode()
X = np.load(str(X_file.absolute()))
y = np.load(str((X_file.parent / X_file.name.replace("x_", "y_")).absolute()))

# filter X features to fit model
target_torque = ['tau_J0', 'tau_J1', 'tau_J2',
                    'tau_J3', 'tau_J4', 'tau_J5', 'tau_J6']
target_position_err = ['e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6']
target_velocity_err = ['de0', 'de1', 'de2', 'de3', 'de4', 'de5', 'de6']
model_features = target_torque + target_position_err + target_velocity_err

dataset_targets = np.load(str((Path(os.environ.get(
    "DATASET_REPO_ROOT_PATH")) / "processedData" / "targets.npy").absolute()))

feature_indices = np.where(np.isin(dataset_targets, model_features))[0]
X = X[:, :, feature_indices]
encoder = LabelEncoder()
y = encoder.fit_transform(y)


#X_test_norm = normalize_window()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_classification = model_classification.to(device)
model_classification.eval()
if model_type == "Transformer":
    X_test_tensor = torch.tensor(np.swapaxes(X, 1, 2), dtype=torch.float32).to(device)
else:
    X_test_tensor = torch.tensor(X, dtype=torch.float32).to(device)

# Perform inference
with torch.no_grad():
    outputs = model_classification(X_test_tensor)
    if model_type == "Transformer":
        predictions = torch.argmax(torch.nn.functional.softmax(outputs, dim=1), dim=1).cpu().numpy()
    else:
        predictions = model_classification.get_predictions(outputs)
    

# Calculate accuracy
accuracy = np.mean(predictions == y)
print(f'Test Accuracy: {accuracy:.4f}')

# Generate confusion matrix
confusion_mat = confusion_matrix(y, predictions)

# Define class labels based on output type
class_labels = ['hard', 'pvc_tube', 'soft']

# Display confusion matrix with labels
try:
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_mat, display_labels=class_labels)
    disp.plot(include_values=True, cmap='Blues',
                ax=None, xticks_rotation='horizontal')
    plt.show()
except:
    print("unable to create confusion matrix due to test set expected labels not matching expected label count")
