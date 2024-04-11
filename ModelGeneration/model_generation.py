import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import find_dotenv, load_dotenv
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
# ===============================================================================================================================================
# on MindLab PC, use the humanObjDetEnv conda environment which has installed all the required dependencies (conda activate humanObjDetEnv)
# ===============================================================================================================================================

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        # self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        # out = self.sigmoid(out)
        out = self.softmax(out)
        return out


def train_val(X, y, k=5, output=1):
    kf = KFold(n_splits=k, shuffle=True)

    epoch_best = 0
    learning_rate_best = 0
    num_layers_best = 0
    hidden_size_best = 0
    current_best_val_score = np.inf
    for epochs in range(100, 101, 50):
        for learning_rate in [0.001]:
            for n_layers in [2,3,4,5,6]:
                for hidden_s in [128,256]:
                    print(
                        f"run with {epochs} epochs, {learning_rate} learning rate, {n_layers} layers, and {hidden_s} hidden units")
                    validation_losses = []
                    accuracies = []
                    for fold, (train_indices, val_indices) in enumerate(kf.split(X)):
                        X_train_fold = X[train_indices]
                        y_train_fold = y[train_indices]
                        X_val_fold = X[val_indices]
                        y_val_fold = y[val_indices]

                        # Define model parameters
                        input_size = 14  # number of features
                        hidden_size = hidden_s  # number of LSTM units
                        num_layers = n_layers  # number of LSTM layers
                        output_size = output  

                        # Create LSTM model
                        model = LSTMModel(input_size, hidden_size,
                                          num_layers, output_size)
                        model = model.to(device)

                        # Convert numpy arrays to PyTorch tensors
                        if output == 1:
                            X_train_tensor = torch.tensor(
                                X_train_fold, dtype=torch.float32).to(device)
                            y_train_tensor = torch.tensor(
                                y_train_fold, dtype=torch.float32).unsqueeze(1).to(device)
                            X_val_tensor = torch.tensor(
                                X_val_fold, dtype=torch.float32).to(device)
                            y_val_tensor = torch.tensor(
                                y_val_fold, dtype=torch.float32).unsqueeze(1).to(device)
                        else:
                            X_train_tensor = torch.tensor(
                                X_train_fold, dtype=torch.float32).to(device)
                            y_train_tensor = torch.tensor(
                                y_train_fold, dtype=torch.long).to(device)
                            X_val_tensor = torch.tensor(
                                X_val_fold, dtype=torch.float32).to(device)
                            y_val_tensor = torch.tensor(
                                y_val_fold, dtype=torch.long).to(device)

                        # Define loss function and optimizer
                        if output == 1:
                            criterion = nn.BCELoss()
                        else:
                            criterion = nn.CrossEntropyLoss()
                        optimizer = optim.Adam(
                            model.parameters(), lr=learning_rate)

                        # Train the model
                        for epoch in range(epochs):
                            optimizer.zero_grad()
                            outputs = model(X_train_tensor)
                            loss = criterion(outputs, y_train_tensor)
                            loss.backward()
                            optimizer.step()

                            # print(f'Epoch [{epoch+1}/{epochs}], Fold [{fold+1}/{k}], Loss: {loss.item():.4f}')

                        # Calculate validation loss
                        with torch.no_grad():
                            val_outputs = model(X_val_tensor)
                            val_loss = criterion(val_outputs, y_val_tensor)
                            if output == 1:
                                predictions = (val_outputs.squeeze() > 0.5).cpu().numpy()
                            else:
                                probabilities = torch.nn.functional.softmax(val_outputs, dim=1)
                                predictions = torch.argmax(probabilities, dim=1).cpu().numpy()

                            # Calculate accuracy
                            accuracies.append(np.mean(predictions == y_val_fold))
                            # print(f'Fold [{fold+1}/{k}], Validation Loss: {val_loss.item():.4f}')
                            validation_losses.append(val_loss.item())

                    average_validation_loss = sum(
                        validation_losses) / len(validation_losses)
                    average_accuracy = sum(
                        accuracies) / len(accuracies)
                    print(f'Average validation loss: {average_validation_loss},average accuracy:{average_accuracy}')
                    if average_validation_loss < current_best_val_score:
                        epoch_best = epochs
                        learning_rate_best = learning_rate
                        num_layers_best = n_layers
                        hidden_size_best = hidden_s
                        current_best_val_score = average_validation_loss

                        print("current best validation loss:",
                              current_best_val_score)

    return epoch_best, learning_rate_best, num_layers_best, hidden_size_best


def train(X_train, y_train, epochs, learning_rate, n_layers, hidden_s, output, file_suffix):
    # Define model parameters
    input_size = 14  # number of features
    hidden_size = hidden_s  # number of LSTM units
    num_layers = n_layers  # number of LSTM layers
    output_size = output  

    # Create LSTM model
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    model = model.to(device)
    
    if output == 1:
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(
            y_train, dtype=torch.float32).unsqueeze(1).to(device)
    else:
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

    # Define loss function and optimizer
    if output == 1:
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_values = []
    # Train the model
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss_values.append(loss.item())
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(),f"ModelGeneration/lstm_model_{file_suffix}.pth")
    epochs_list = list(range(1, epochs + 1))
    make_plot(epochs_list, loss_values, file_suffix)


def make_plot(x, y, file_suffix):
    plt.plot(x, y, 'bo', label='Training loss')
    plt.title(f'Training loss over epochs for {file_suffix}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Save the plot to an image file
    plt.savefig(f"ModelGeneration/training_loss_{file_suffix}.png")

    # Close the plot to free up memory
    plt.close()


def evaluate(X_test, y_test, hidden_s, n_layers, output, file_suffix):
    # Load the trained model
    model = LSTMModel(input_size=14, hidden_size=hidden_s,
                      num_layers=n_layers, output_size=output)
    model.load_state_dict(torch.load(
        f"ModelGeneration/lstm_model_{file_suffix}.pth"))
    model.eval()

    # Convert numpy arrays to PyTorch tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Perform inference
    with torch.no_grad():
        outputs = model(X_test_tensor)
    
    if output == 1:
        predictions = (outputs.squeeze() > 0.5).cpu().numpy()
    else:
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1).cpu().numpy()

    # Calculate accuracy
    accuracy = np.mean(predictions == y_test)

    # Generate confusion matrix
    confusion_mat = confusion_matrix(y_test, predictions)

    # Define class labels based on output type
    if output == 1:
        class_labels = ['Negative', 'Positive']
    else:
        # change mapper according to encoding of y
        mapper = {0: 'hard', 1: 'plasticbottle', 2: 'soft'}
        class_labels = [mapper[i] for i in range(output)]

    print(f'Test Accuracy: {accuracy:.4f}')
    # Display confusion matrix with labels
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_mat, display_labels=class_labels)
    disp.plot(include_values=True, cmap='Blues',
              ax=None, xticks_rotation='horizontal')
    plt.show()

def save_params(model_name,epochs,learning_rate,num_layers,hidden_size):
    best_params = {
        "model name":model_name,
        "epochs":epochs,
        "learning rate": learning_rate,
        "number of layers":num_layers,
        "hidden size": hidden_size,
        'modification_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    file_path = f"ModelGeneration/parameter_file.txt"
   
    mode = 'a' if os.path.exists(file_path) else 'w'
    
    with open(file_path, mode) as f:
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

if __name__ == '__main__':
    load_dotenv(find_dotenv())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Print the device being used
    print('Using device:', device)

    # specify dataset-files (via file suffix)
    # dataset with a single extracted time-window per contact, beginning at exact first contact time
    #files_suffix = "single_on_contact"

    # dataset with one extracted time-window per contact, beginning 100ms (= 20 robot data rows) before first contact time
    files_suffix = "sliding_left_offset20240410_c4"

    # dataset with multiple extracted (sliding) time-windows per contact, beginning 100ms before contact time, until end of contact is reached
    # sliding window step is 4 robot data rows = 20ms
    #files_suffix = "sliding_left_offset"

    X_file, y_file = f"x_{files_suffix}.npy", f"y_{files_suffix}.npy"

    processed_data_folder_path = Path(os.environ.get(
        "DATASET_REPO_ROOT_PATH")) / "processedData"
    X = np.load(str((processed_data_folder_path / X_file).absolute()))
    y = np.load(str((processed_data_folder_path / y_file).absolute()))

    # filter X features to fit model
    # (as of 02.04.2024) all datasets contain the following features in that order (torque- / position- / velocity errors):
    # ['etau_J0', 'etau_J1', 'etau_J2', 'etau_J3', 'etau_J4', 'etau_J5', 'etau_J6', 'e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'de0', 'de1', 'de2', 'de3', 'de4', 'de5', 'de6']
    X = X[:, :, 0:14]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.1)

    
    
    #X_train_normalized = (X_train - X_train.mean(axis=2, keepdims=True)) / X_train.std(axis=2,keepdims=True)
    #X_test_normalized = (X_test- X_test.mean(axis=2, keepdims=True)) / X_test.std(axis=2,keepdims=True)
    X_train_normalized = (X_train - X_train.min(axis=2, keepdims=True)) / (X_train.max(axis=2,keepdims=True)-X_train.min(axis=2, keepdims=True))
    X_test_normalized = (X_test - X_test.min(axis=2, keepdims=True)) / (X_test.max(axis=2,keepdims=True)-X_test.min(axis=2, keepdims=True))
    X_train = X_train_normalized
    X_test = X_test_normalized
    
    # k-fold cross validation
    e, lr, num_layers, hidden_size = train_val(X_train, y_train, k=5, output=3)
    print(f"Best score result is a combination of {e} epochs, learning rate of {lr}, {num_layers} number of layers and a hidden size of {hidden_size}.")
    files_suffix = "sliding_left_offset20240410_c4_norm"
    save_params(files_suffix,e,lr,num_layers,hidden_size)
    
    # train the model 
    train(X_train, y_train, epochs=e, learning_rate=lr,n_layers=num_layers, hidden_s=hidden_size, output=3, file_suffix=files_suffix)
    # evaluate the trained model
    evaluate(X_test, y_test, n_layers=num_layers, hidden_s=hidden_size,output=3, file_suffix=files_suffix)  
