import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import json
from datetime import datetime
from pathlib import Path
from typing import Type

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from dotenv import find_dotenv, load_dotenv
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from ModelGeneration.earlystopping import EarlyStopper
from ModelGeneration.rnn_models import (GRUModel, LSTMModel, RNNModel, GRUModelWithLayerNorm, LSTMModelWithLayerNorm,
                                        RNNModelHyperParameters,
                                        RNNModelHyperParameterSet)


# ===========================================================================================================================================================
# on MindLab PC, use the humanObjectDetectionEnv conda environment which has installed all the required dependencies (conda activate humanObjectDetectionEnv)
# ===========================================================================================================================================================

model_classes: "list[Type[RNNModel]]" = [LSTMModel,
                                         LSTMModelWithLayerNorm, GRUModel, GRUModelWithLayerNorm]


hidden_sizes = [16,32,64,128]#, 256]
num_layers = [1, 2, 3]#, 4]
epochs = np.arange(100, 201, 50)
learning_rates = [0.001, 0.01,0.0001,0.1]
input_size = 21
output_size = 3


def get_trained_models_path():
    return Path(__file__).parent.absolute() / "TrainedModels"


def get_model_params_path(model_prefix, file_suffix):
    trained_models_path = get_trained_models_path()
    return str((trained_models_path / f"{model_prefix}_{file_suffix}.pth").absolute())


class RNNModelTrainer():
    def __init__(self, device, model_class: Type[RNNModel], hyperparameters: RNNModelHyperParameters, X_train, y_train, X_test, y_test, optimizer="Adam"):
        self.device = device
        self.model_class = model_class
        self.hyperparameters = hyperparameters
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.optimizer = optimizer

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
                if self.optimizer == "AdamW":
                    optimizer = optim.AdamW(
                        model.parameters(), lr=hp.learning_rate, weight_decay=0.01)
                else:
                    optimizer = optim.Adam(
                        model.parameters(), lr=hp.learning_rate)

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
        stopper = EarlyStopper()
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
        if self.optimizer == "AdamW":
            optimizer = optim.AdamW(model.parameters(
            ), lr=self.hyperparameters.best_hyperparameters.learning_rate, weight_decay=0.01)
        else:
            optimizer = optim.Adam(
                model.parameters(), lr=self.hyperparameters.best_hyperparameters.learning_rate)

        loss_values = []
        # Train the model
        for epoch in range(self.hyperparameters.best_hyperparameters.epochs):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            if stopper.early_stop(loss.item()):
                self.hyperparameters.best_hyperparameters.epochs = epoch
                break
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
            'hard', 'pvc_tube', 'soft']

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


def save_hyperparameters(model_name, hyperparameters: RNNModelHyperParameterSet, optimizer,mean,std):
    model_params_list = []
    file_path = get_trained_models_path() / "RnnModelsParameters.json"
    print(file_path)
    if os.path.exists(str(file_path.absolute())):
        with open(str(file_path.absolute()), 'r') as f:
            model_params_list = json.load(f)

    model_params_list = [
        i for i in model_params_list if i["model_name"] != model_name]
    new_params = {
        'model_name': model_name,
        'modification_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'hyperparameters': {
            "hidden_size": int(hyperparameters.hidden_size),
            "num_layers": int(hyperparameters.num_layers),
            "epochs": int(hyperparameters.epochs),
            "learning_rate": float(hyperparameters.learning_rate)
        },
        'optimizer': optimizer,    
    }

    if (len(mean) != 0) and (len(std)!=0):
        new_params['normalization_mean'] = mean
        new_params['normalization_std'] = std

    model_params_list.append(new_params)

    with open(str(file_path.absolute()), 'w') as f:
        json.dump(model_params_list, f, indent=4)


def choose_model_class() -> Type[RNNModel]:
    lines = [f'{i} {t.__name__}' for i, t in enumerate(model_classes)]
    print("RNN Model Classes:")
    print('\n'.join(lines) + '\n')
    model_class_index = None
    while model_class_index not in np.arange(0, len(model_classes), 1):
        model_class_index = int(input(
            "Which model class should be used? (choose by index): "))
    return model_classes[model_class_index]


def choose_dataset():
    processed_data_path = Path(os.environ.get(
        "DATASET_REPO_ROOT_PATH")) / "processedData"
    
    sub_repo = dict([(str(i),p) for i,p in enumerate(processed_data_path.iterdir()) if p.is_dir()])
    print("sub repo:")
    lines = [f'{key} {value.name}' for key, value in sub_repo.items()]
    print('\n'.join(lines) + '\n')
    subrepo_key = None
    while subrepo_key not in sub_repo:
        subrepo_key = input(
            "Which sub repo should be used? (choose by index): ")

    full_path = processed_data_path / sub_repo[subrepo_key]

    datasets = dict([(str(i), p) for i, p in enumerate(full_path.iterdir())
                     if p.is_file and p.name.startswith("x_") and p.suffix == ".npy" and "test" not in p.name])
    lines = [f'{key} {value.name}' for key, value in datasets.items()]
    
    print("Datasets:")
    print('\n'.join(lines) + '\n')
    dataset_key = None
    while dataset_key not in datasets:
        dataset_key = input(
            "Which dataset should be used? (choose by index): ")
    return sub_repo[subrepo_key],datasets[dataset_key]


def choose_normalization_mode():
    normalization_choice = ""
    while normalization_choice not in ["y", "n"]:
        normalization_choice = input(
            "Should the data be normalized? (y / n): ").lower()
    return True if normalization_choice == "y" else False


def choose_optimizer():
    optimizer_choice = ""
    while optimizer_choice not in ["0", "1"]:
        optimizer_choice = input(
            "Which optimizer do you want to use? (Adam=0 / AdamW=1): ").lower()
    return "AdamW" if optimizer_choice == "1" else "Adam"


if __name__ == '__main__':
    load_dotenv(find_dotenv())

    model_class = choose_model_class()
    sub_repo, X_file = choose_dataset()
    normalize = choose_normalization_mode()
    optimizer = choose_optimizer()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    X = np.load(str(X_file.absolute()))
    y = np.load(
        str((X_file.parent / X_file.name.replace("x_", "y_")).absolute()))

    # filter X features to fit model
    torque_indices = np.arange(0,7, 1)
    position_error_indices = np.arange(28, 35, 1)
    velocity_error_indices = np.arange(35,42, 1)
    feature_indices = np.concatenate((torque_indices, position_error_indices, velocity_error_indices))
    X = X[:, :, feature_indices]
    encoder = LabelEncoder()
    if sub_repo != 'test_train_split':
        X_train, X_test, y_train, y_test = train_test_split(
            X, encoder.fit_transform(y), test_size=0.1)
        files_suffix = X_file.name.replace("x_", "").replace(".npy", "")
    else:
        X_train = X
        y_train = encoder.fit_transform(y)
        X_test = np.load(str((X_file.parent / X_file.name.replace("train", "test")).absolute()))
        X_test = X_test[:,:,feature_indices]
        y_test = np.load(str((X_file.parent / X_file.name.replace("x_train", "y_test")).absolute()))
        y_test = encoder.transform(y_test)
        files_suffix = X_file.name.replace("x_train", "split").replace(".npy", "")

    
    mean = []
    std = []
    if normalize:
        for i in range(X_train.shape[2]):
            scaler = StandardScaler()
            X_train[:, :, i] = scaler.fit_transform(X_train[:, :, i])
            X_test[:, :, i] = scaler.transform(X_test[:, :, i])
            mean.append((scaler.mean_).tolist())
            std.append((scaler.scale_).tolist())
        files_suffix += "_norm"
    

    rnn_model_trainer = RNNModelTrainer(
        device=device,
        model_class=model_class,
        hyperparameters=RNNModelHyperParameters(
            hidden_sizes=hidden_sizes, num_layers=num_layers, epochs=epochs,
            learning_rates=learning_rates, input_size=input_size, output_size=output_size),
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, optimizer=optimizer)

    # k-fold cross validation for hyperparameters
    rnn_model_trainer.kFold_cross_validate(k=5)
    print(
        f"""Cross-Validation best score reached for:
            {rnn_model_trainer.hyperparameters.best_hyperparameters.epochs} epochs,
            {rnn_model_trainer.hyperparameters.best_hyperparameters.learning_rate} learning rate,
            {rnn_model_trainer.hyperparameters.best_hyperparameters.num_layers} number of layers,
            {rnn_model_trainer.hyperparameters.best_hyperparameters.hidden_size} hidden size""")

    save_hyperparameters(
        f"{model_class.__name__}_{files_suffix}", rnn_model_trainer.hyperparameters.best_hyperparameters, rnn_model_trainer.optimizer,mean,std)

    # train and evaluate the model
    rnn_model_trainer.train_model(file_suffix=files_suffix)
    rnn_model_trainer.evaluate_model(file_suffix=files_suffix)
