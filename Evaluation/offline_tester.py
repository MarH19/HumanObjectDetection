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
                        load_transformer_classification_model,
                        normalize_window)
from ModelGeneration.SoftVotingClassifier import SoftVotingClassifier
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


_, X_file = choose_dataset()

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
y[:, 0] = encoder.fit_transform(y[:, 0])

# normalize
for i, x_i in enumerate(X):
    X[i] = normalize_window(
        x_i, rnn_model_params if model_type == "RNN" else transformer_config)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_classification = model_classification.to(device)
model_classification.eval()

if model_type == "Transformer":
    X_test_tensor = torch.tensor(np.swapaxes(
        X, 1, 2), dtype=torch.float32).to(device)
else:
    X_test_tensor = torch.tensor(X, dtype=torch.float32).to(device)

# initialize soft voting classifier
softvoting_window_lengths = [8, 10, 12, 15]
soft_voting_classifiers = [SoftVotingClassifier(
    model_classification, swl) for swl in softvoting_window_lengths]

# group X data by contact to simulate individual contact predictions for soft voting
unique_inst_contact_ids = np.unique(y[:, 1])
X_grouped_by_contact = [X_test_tensor[y[:, 1] == i]
                        for i in unique_inst_contact_ids]
y_by_contact = np.array([int(y[y[:, 1] == i][0][0])
                        for i in unique_inst_contact_ids])

# Perform inference with soft voting (majority vote)
soft_voting_predictions = [[] for _ in range(len(softvoting_window_lengths))]
for i, soft_voter in enumerate(soft_voting_classifiers):
    for X_contact_tensor in X_grouped_by_contact:
        contact_predictions = []
        for X_window_at_timestep in X_contact_tensor.unsqueeze(1):
            contact_predictions.append(
                soft_voter.predict(X_window_at_timestep))
        soft_voting_predictions[i].append(contact_predictions)
        soft_voter.reset()

# assert that for each soft voting prediction per contact, only the prediction at the given soft voting window length index is not None
# assert that for each soft voting prediction per contact, the prediction at the given soft voting window length index is a proper class label
for i, predictions in enumerate(soft_voting_predictions):
    for j, contact_predictions in enumerate(predictions):
        assert all(value is None for k, value in enumerate(
            contact_predictions) if k != softvoting_window_lengths[i] - 1), f"soft voting with window length {softvoting_window_lengths[i]} made a prediction at a timestep other than the window length index"
        assert contact_predictions[softvoting_window_lengths[i] - 1] in [
            0, 1, 2], f"soft voting with window length {softvoting_window_lengths[i]} made a prediction that is not a proper class label"

        # replace array of soft voting predictions with the prediction at the given window length index -> the one actual prediction of the soft voting classifier
        soft_voting_predictions[i][j] = [
            p for p in contact_predictions if p is not None][0]

soft_voting_predictions = np.array(soft_voting_predictions)

# Calculate accuracies for all soft voting classifiers
accuracies = [{"caption": f"soft voting classifier that evaluates first {swl} predictions per contact", "accuracy": np.mean(
    soft_voting_predictions[i] == y_by_contact)} for i, swl in enumerate(softvoting_window_lengths)]

print(f"\nWith window size {window_classification_length}")
for accuracy in accuracies:
    print(
        f'Test Accuracy for {accuracy["caption"]}: {accuracy["accuracy"]:.4f}')

sys.exit()
# Generate confusion matrix
highest_accuracy_item = max(accuracies, key=lambda x: x["accuracy"])
confusion_mat = confusion_matrix(y[highest_accuracy_item["indices"], 0].astype(
    int), predictions[highest_accuracy_item["indices"]])

# Define class labels based on output type
class_labels = ['hard', 'pvc_tube', 'soft']

# Display confusion matrix with labels
try:
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_mat, display_labels=class_labels)
    disp.plot(include_values=True, cmap='Blues',
              ax=None, xticks_rotation='horizontal')
    plt.title(f'predictions {highest_accuracy_item["caption"]}')
    plt.show()
except:
    print("unable to create confusion matrix due to test set expected labels not matching expected label count")
