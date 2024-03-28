import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# comment Justin: LSTM seems like a good fit, maybe GRU would be worth a try to avoid overfitting (since they're simpler)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        #self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        #out = self.sigmoid(out)
        out = self.softmax(out)
        return out


def train_val(X, y, k=5,output=1):
    kf = KFold(n_splits=k, shuffle=True)

    epoch_best = 0
    learning_rate_best = 0
    num_layers_best = 0
    hidden_size_best = 0
    current_best_val_score = np.inf
    for epochs in range(50, 101, 50):
        for learning_rate in [0.001, 0.01, 0.1]:
            for n_layers in [1, 2, 3, 4]:
                for hidden_s in [32, 64, 128]: 
                    print(f"run with {epochs} epochs, {learning_rate} learning rate, {n_layers} layers, and {hidden_s} hidden units")
                    validation_losses = []
                    for fold, (train_indices, val_indices) in enumerate(kf.split(X)):
                        X_train_fold = X[train_indices]
                        y_train_fold = y[train_indices]
                        X_val_fold = X[val_indices]
                        y_val_fold = y[val_indices]

                        # Define model parameters
                        # comment Justin: would it be worth a try to also tune the hidden_size and num_layers via k-fold val.?
                        input_size = 14  # number of features
                        hidden_size = hidden_s  # number of LSTM units
                        num_layers = n_layers  # number of LSTM layers
                        output_size = output  # binary classification

                        # Create LSTM model
                        model = LSTMModel(input_size, hidden_size,
                                        num_layers, output_size)
                        model = model.to(device)

                        # Convert numpy arrays to PyTorch tensors
                        if output ==1:
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
                        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
                            # print(f'Fold [{fold+1}/{k}], Validation Loss: {val_loss.item():.4f}')
                            validation_losses.append(val_loss.item())

                    average_validation_loss = sum(
                        validation_losses) / len(validation_losses)
                    # print(f'Average validation loss: {average_validation_loss}')
                    if average_validation_loss < current_best_val_score:
                        epoch_best = epochs
                        learning_rate_best = learning_rate
                        num_layers_best = n_layers
                        hidden_size_best = hidden_s
                        current_best_val_score = average_validation_loss

                        print("current best validation score:", current_best_val_score)

    return epoch_best, learning_rate_best, num_layers_best, hidden_size_best


def train(X_train, y_train, epochs, learning_rate,n_layers,hidden_s,output):
    # Define model parameters
    input_size = 14  # number of features
    hidden_size = hidden_s  # number of LSTM units
    num_layers = n_layers  # number of LSTM layers
    output_size = output  # binary classification

    # Create LSTM model
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    model = model.to(device)
    # Convert numpy arrays to PyTorch tensors
    if output ==1:
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
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

    torch.save(model.state_dict(),
               'ModelGeneration/lstm_model_c4hard.pth')
    epochs_list = list(range(1, epochs + 1))
    make_plot(epochs_list, loss_values)


def make_plot(x, y):
    plt.plot(x, y, 'bo', label='Training loss')
    plt.title('Training loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Save the plot to an image file
    plt.savefig('ModelGeneration/training_loss_plot.png')

    # Close the plot to free up memory
    plt.close()


def evaluate(X_test, y_test,hidden_s,n_layers,output):
    # Load the trained model
    model = LSTMModel(input_size=14, hidden_size=hidden_s,
                      num_layers=n_layers, output_size=output)
    model.load_state_dict(torch.load(
        'ModelGeneration/lstm_model_c4hard.pth'))
    model.eval()

    # Convert numpy arrays to PyTorch tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Perform inference
    with torch.no_grad():
        outputs = model(X_test_tensor)
        # Convert to binary predictions
    if output ==1:
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
        mapper = {0: 'hard', 1: 'plasticbottle', 2: 'soft'} # change mapper according to encoding of y
        class_labels = [mapper[i] for i in range(output)]
        

    print(f'Test Accuracy: {accuracy:.4f}')
    # Display confusion matrix with labels
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=class_labels)
    disp.plot(include_values=True, cmap='Blues', ax=None, xticks_rotation='horizontal')
    plt.show()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Print the device being used
    print('Using device:', device)

    #folder_path = Path('/home/mindlab/humanObjectDetectionDataset/processedData')
    #X = np.load(str((folder_path / "x_data.npy").absolute()))
    #y = np.load(str((folder_path / "y_data.npy").absolute()))
    X = np.load("DataPreparation/x_data_a4_torqueposition.npy")
    y = np.load("DataPreparation/y_data_a4_torqueposition.npy")
 
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2)

    # comment Justin: ToDo: normalize data (need to do this here instead of in data preparation,
    # because need to calculate normalization values only on training set and the apply same values to test set)
    e, lr,num_layers,hidden_size = train_val(X_train, y_train, k=5,output=3)  # k-fold cross validation
    print(f"Best score result is a combination of {e} epochs, learning rate of {lr}, {num_layers} number of layers and a hidden size of {hidden_size}.")
    # train the model with the whole dataset
    train(X_train, y_train, epochs=e, learning_rate=lr,n_layers=num_layers,hidden_s=hidden_size,output=3)
    evaluate(X_test, y_test,n_layers=num_layers,hidden_s=hidden_size,output=3)  # evaluate the trained model
