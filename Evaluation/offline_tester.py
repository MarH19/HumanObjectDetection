import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from _util.util import (choose_dataset, choose_model_type,
                        choose_trained_rnn_model,
                        choose_trained_transformer_model,
                        load_rnn_classification_model,
                        load_transformer_classification_model)
from ModelGeneration.model_generation import choose_rnn_model_class

classification_model_input_size = 21
window_classification_length = 40
labels_classification = {0: "hard", 1: "pvc_tube", 2: "soft"}


load_dotenv(find_dotenv())

model_type = choose_model_type()
model_classification, rnn_model_params, transformer_config = None, None, None
if model_type == "RNN":
    rnn_model_class = choose_rnn_model_class()
    rnn_model_params = choose_trained_rnn_model(rnn_model_class)
    model_classification = load_rnn_classification_model(
        rnn_model_class, rnn_model_params, classification_model_input_size, len(
            labels_classification))
elif model_type == "Transformer":
    transformer_model_path = choose_trained_transformer_model()
    model_classification, transformer_config = load_transformer_classification_model(
        transformer_model_path, classification_model_input_size, len(labels_classification), window_classification_length)


sub_repo, X_file = choose_dataset()

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

# X_test_norm = normalize_window()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_classification = model_classification.to(device)
model_classification.eval()

if model_type == "Transformer":
    X_test_tensor = torch.tensor(np.swapaxes(
        X, 1, 2), dtype=torch.float32).to(device)
else:
    X_test_tensor = torch.tensor(X, dtype=torch.float32).to(device)

# Perform inference
with torch.no_grad():
    outputs = model_classification(X_test_tensor)
    if model_type == "Transformer":
        predictions = torch.argmax(torch.nn.functional.softmax(
            outputs, dim=1), dim=1).cpu().numpy()
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
