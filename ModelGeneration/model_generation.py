import os
from datetime import datetime
from pathlib import Path
from typing import Type

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import find_dotenv, load_dotenv
from rnn_models import LSTMModel, RNNModel, RNNModelHyperParameters, RNNModelHyperParameterSet
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder

# ===============================================================================================================================================
# on MindLab PC, use the humanObjectDetectionEnv conda environment which has installed all the required dependencies (conda activate humanObjDetEnv)
# ===============================================================================================================================================


def get_trained_models_path():
    return Path(__file__).parent.absolute() / "TrainedModels"


def get_model_params_path(model_prefix, file_suffix):
    trained_models_path = get_trained_models_path()
    return str((trained_models_path / f"{model_prefix}_{file_suffix}.pth").absolute())


class RNNModelTrainer():
    def __init__(self, device, model_class: Type[RNNModel], hyperparameters: RNNModelHyperParameters, X_train, y_train, X_test, y_test):
        self.device = device
        self.model_class = model_class
        self.hyperparameters = hyperparameters
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_kFold_cross_validation_tensors(self, train_indices, val_indices):
        X_train_fold = self.X_train[train_indices]
        y_train_fold = self.y_train[train_indices]
        X_val_fold = self.X_train[val_indices]
        y_val_fold = self.y_train[val_indices]

        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.tensor(
            X_train_fold, dtype=torch.float32).to(self.device)
        X_val_tensor = torch.tensor(
            X_val_fold, dtype=torch.float32).to(self.device)

        if self.hyperparameters.output_size == 1:
            y_train_tensor = torch.tensor(
                y_train_fold, dtype=torch.float32).unsqueeze(1).to(self.device)
            y_val_tensor = torch.tensor(
                y_val_fold, dtype=torch.float32).unsqueeze(1).to(self.device)
        else:
            y_train_tensor = torch.tensor(
                y_train_fold, dtype=torch.long).to(self.device)
            y_val_tensor = torch.tensor(
                y_val_fold, dtype=torch.long).to(self.device)

        return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor

    def kFold_cross_validate(self, k=5):
        kf = KFold(n_splits=k, shuffle=True)

        current_best_val_score = np.inf
        for hp in self.hyperparameters.get_hyperparameter_combinations():
            print(
                f"cross-validation with {hp.epochs} epochs, {hp.learning_rate} learning rate, {hp.num_layers} layers, and {hp.hidden_size} hidden units")

            validation_losses = []
            accuracies = []
            for _, (train_indices, val_indices) in enumerate(kf.split(self.X_train)):
                # get split tensors
                X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor = self.get_kFold_cross_validation_tensors(
                    train_indices=train_indices, val_indices=val_indices)

                # instantiate RNN model
                model = self.model_class(input_size=self.hyperparameters.input_size, hidden_size=hp.hidden_size,
                                         num_layers=hp.num_layers, output_size=self.hyperparameters.output_size)
                model = model.to(self.device)

                # Define loss function and optimizer
                criterion = model.get_criterion()
                optimizer = optim.Adam(model.parameters(), lr=hp.learning_rate)

                # Train the model
                for _ in range(hp.epochs):
                    optimizer.zero_grad()
                    outputs = model(X_train_tensor)
                    loss = criterion(outputs, y_train_tensor)
                    loss.backward()
                    optimizer.step()

                # Calculate validation loss and accuracy
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor)
                    validation_losses.append(val_loss.item())
                    predictions = model.get_predictions(val_outputs)
                    accuracies.append(
                        np.mean(predictions == y_val_tensor.cpu().numpy()))

            average_validation_loss = sum(
                validation_losses) / len(validation_losses)
            average_accuracy = sum(
                accuracies) / len(accuracies)
            print(
                f'Average validation loss: {average_validation_loss}, average accuracy:{average_accuracy}')

            if average_validation_loss < current_best_val_score:
                self.hyperparameters.best_hyperparameters = hp
                current_best_val_score = average_validation_loss

                print(
                    f"current best validation loss: {current_best_val_score}")

    def train_model(self, file_suffix):
        # Create LSTM model
        model = self.model_class(self.hyperparameters.input_size, self.hyperparameters.best_hyperparameters.hidden_size,
                                 self.hyperparameters.best_hyperparameters.num_layers, self.hyperparameters.output_size)
        model = model.to(self.device)

        X_train_tensor = torch.tensor(
            self.X_train, dtype=torch.float32).to(self.device)
        if self.hyperparameters.output_size == 1:
            y_train_tensor = torch.tensor(
                self.y_train, dtype=torch.float32).unsqueeze(1).to(self.device)
        else:
            y_train_tensor = torch.tensor(
                self.y_train, dtype=torch.long).to(self.device)

        # Define loss function and optimizer
        criterion = model.get_criterion()
        optimizer = optim.Adam(
            model.parameters(), lr=self.hyperparameters.best_hyperparameters.learning_rate)

        loss_values = []
        # Train the model
        for epoch in range(self.hyperparameters.best_hyperparameters.epochs):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss_values.append(loss.item())
            loss.backward()
            optimizer.step()

            print(
                f'Epoch [{epoch+1}/{self.hyperparameters.best_hyperparameters.epochs}], Loss: {loss.item():.4f}')

        torch.save(model.state_dict(), get_model_params_path(
            model_prefix=self.model_class.__name__, file_suffix=file_suffix))
        plot_model_training_loss(
            self.hyperparameters.best_hyperparameters.epochs, loss_values, self.model_class, file_suffix)

    def evaluate_model(self, file_suffix):
        # Load the trained model
        model = self.model_class(input_size=self.hyperparameters.input_size, hidden_size=self.hyperparameters.best_hyperparameters.hidden_size,
                                 num_layers=self.hyperparameters.best_hyperparameters.num_layers, output_size=self.hyperparameters.output_size)
        model.load_state_dict(torch.load(get_model_params_path(
            model_prefix=self.model_class.__name__, file_suffix=file_suffix)))
        model.eval()

        # Convert numpy arrays to PyTorch tensors
        X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32)

        # Perform inference
        with torch.no_grad():
            outputs = model(X_test_tensor)
            predictions = model.get_predictions(outputs)

        # Calculate accuracy
        accuracy = np.mean(predictions == self.y_test)
        print(f'Test Accuracy: {accuracy:.4f}')

        # Generate confusion matrix
        confusion_mat = confusion_matrix(self.y_test, predictions)

        # Define class labels based on output type
        class_labels = ['Negative', 'Positive'] if self.hyperparameters.output_size == 1 else [
            'hard', 'plasticbottle', 'soft']

        # Display confusion matrix with labels
        try:
            disp = ConfusionMatrixDisplay(
                confusion_matrix=confusion_mat, display_labels=class_labels)
            disp.plot(include_values=True, cmap='Blues',
                      ax=None, xticks_rotation='horizontal')

            # save confusion matrix plot
            trained_model_results_path = get_trained_models_path() / "Results"
            disp.figure_.savefig(str(
                (trained_model_results_path / f"confusion_matrix_{self.model_class.__name__}_{file_suffix}.png").absolute()))

            plt.show()
        except:
            print("unable to create confusion matrix due to test set expected labels not matching expected label count")


def plot_model_training_loss(epochs, loss_values, model_class: Type[RNNModel], file_suffix):
    plt.plot(list(range(1, epochs + 1)), loss_values, label='Training loss')
    plt.title(f'Training loss over epochs for {file_suffix}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Save the plot to an image file
    trained_model_results_path = get_trained_models_path() / "Results"
    plt.savefig(str((trained_model_results_path /
                f"training_loss_{model_class.__name__}_{file_suffix}.png").absolute()))

    # Close the plot to free up memory
    plt.close()


def save_hyperparameters(model_name, hyperparameters: RNNModelHyperParameterSet):
    best_params = {
        "model name": model_name,
        "epochs": hyperparameters.epochs,
        "learning rate": hyperparameters.learning_rate,
        "number of layers": hyperparameters.num_layers,
        "hidden size": hyperparameters.hidden_size,
        'modification_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    file_path = get_trained_models_path() / "parameter_file.txt"
    mode = 'a' if os.path.exists(str(file_path.absolute())) else 'w'
    with open(str(file_path.absolute()), mode) as f:
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")


if __name__ == '__main__':
    load_dotenv(find_dotenv())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # specify dataset-files (via file suffix)
    # dataset with a single extracted time-window per contact, beginning at exact first contact time
    # files_suffix = "single_on_contact"
    # files_suffix = "single_on_contact20240410_c4"

    # dataset with one extracted time-window per contact, beginning 100ms (= 20 robot data rows) before first contact time
    files_suffix = "single_left_offset"

    # dataset with multiple extracted (sliding) time-windows per contact, beginning 100ms before contact time, until end of contact is reached
    # sliding window step is 4 robot data rows = 20ms
    # files_suffix = "sliding_left_offset"
    # files_suffix = "sliding_left_offset20240410_c4"

    X_file, y_file = f"x_{files_suffix}.npy", f"y_{files_suffix}.npy"

    processed_data_folder_path = Path(os.environ.get(
        "DATASET_REPO_ROOT_PATH")) / "processedData"
    X = np.load(str((processed_data_folder_path / X_file).absolute()))
    y = np.load(str((processed_data_folder_path / y_file).absolute()))

    # filter X features to fit model
    # (as of 02.04.2024) all datasets contain the following features in that order (torque- / position- / velocity errors):
    # ['etau_J0', 'etau_J1', 'etau_J2', 'etau_J3', 'etau_J4', 'etau_J5', 'etau_J6', 'e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'de0', 'de1', 'de2', 'de3', 'de4', 'de5', 'de6']
    X = X[:, :, 0:14]

    X_train, X_test, y_train, y_test = train_test_split(
        X, LabelEncoder().fit_transform(y), test_size=0.1)

    normalize = False
    if normalize:
        X_train = (X_train - X_train.min(axis=2, keepdims=True)) / \
            (X_train.max(axis=2, keepdims=True)-X_train.min(axis=2, keepdims=True))
        X_test = (X_test - X_test.min(axis=2, keepdims=True)) / \
            (X_test.max(axis=2, keepdims=True)-X_test.min(axis=2, keepdims=True))
        files_suffix += "_norm"

    model_class = LSTMModel
    rnn_model_trainer = RNNModelTrainer(device=device, model_class=model_class, hyperparameters=RNNModelHyperParameters(
        hidden_sizes=[32, 64, 128], num_layers=[1, 2], epochs=[100, 150], learning_rates=[0.001, 0.01, 0.1], input_size=14, output_size=3),
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    # k-fold cross validation for hyperparameters
    rnn_model_trainer.kFold_cross_validate(k=5)
    print(
        f"""Cross-Validation best score reached for:
            {rnn_model_trainer.hyperparameters.best_hyperparameters.epochs} epochs,
            {rnn_model_trainer.hyperparameters.best_hyperparameters.learning_rate} learning rate,
            {rnn_model_trainer.hyperparameters.best_hyperparameters.num_layers} number of layers,
            {rnn_model_trainer.hyperparameters.best_hyperparameters.hidden_size} hidden size""")

    save_hyperparameters(
        f"{model_class.__name__}_{files_suffix}", rnn_model_trainer.hyperparameters.best_hyperparameters)

    # train and evaluate the model
    rnn_model_trainer.train_model(file_suffix=files_suffix)
    rnn_model_trainer.evaluate_model(file_suffix=files_suffix)
