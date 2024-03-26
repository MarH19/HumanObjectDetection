import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# comment Justin: LSTM seems like a good fit, maybe GRU would be worth a try to avoid overfitting (since they're simpler)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out


def train_val(X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True)

    epoch_best = 0
    learning_rate_best = 0
    current_best_val_score = np.inf
    for epochs in range(100, 351, 50):
        for learning_rate in [0.001, 0.01, 0.1]:
            print(
                f"run with {epochs} epochs and {learning_rate} learning rate")
            validation_losses = []
            for fold, (train_indices, val_indices) in enumerate(kf.split(X)):
                X_train_fold = X[train_indices]
                y_train_fold = y[train_indices]
                X_val_fold = X[val_indices]
                y_val_fold = y[val_indices]

                # Define model parameters
                # comment Justin: would it be worth a try to also tune the hidden_size and num_layers via k-fold val.?
                input_size = 7  # number of features
                hidden_size = 64  # number of LSTM units
                num_layers = 1  # number of LSTM layers
                output_size = 1  # binary classification

                # Create LSTM model
                model = LSTMModel(input_size, hidden_size,
                                  num_layers, output_size)
                model = model.to(device)

                # Convert numpy arrays to PyTorch tensors
                X_train_tensor = torch.tensor(
                    X_train_fold, dtype=torch.float32).to(device)
                y_train_tensor = torch.tensor(
                    y_train_fold, dtype=torch.float32).unsqueeze(1).to(device)
                X_val_tensor = torch.tensor(
                    X_val_fold, dtype=torch.float32).to(device)
                y_val_tensor = torch.tensor(
                    y_val_fold, dtype=torch.float32).unsqueeze(1).to(device)

                # Define loss function and optimizer
                criterion = nn.BCELoss() 
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
                current_best_val_score = average_validation_loss
                print("current best validation score:", current_best_val_score)

    return epoch_best, learning_rate_best


def train(X_train, y_train, epochs, learning_rate):
    # Define model parameters
    input_size = 7  # number of features
    hidden_size = 64  # number of LSTM units
    num_layers = 1  # number of LSTM layers
    output_size = 1  # binary classification

    # Create LSTM model
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    model = model.to(device)
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(
        y_train, dtype=torch.float32).unsqueeze(1).to(device)

    # Define loss function and optimizer
    # TODO change criterion when not binary anymore
    criterion = nn.BCELoss()
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
               'ModelGeneration/lstm_model_all_withoutc4hard.pth')
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


def evaluate(X_test, y_test):
    # Load the trained model
    model = LSTMModel(input_size=7, hidden_size=64,
                      num_layers=1, output_size=1)
    model.load_state_dict(torch.load(
        'ModelGeneration/lstm_model_all_withoutc4hard.pth'))
    model.eval()

    # Convert numpy arrays to PyTorch tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Perform inference
    with torch.no_grad():
        outputs = model(X_test_tensor)
        # Convert to binary predictions
        predictions = (outputs.squeeze() > 0.5).cpu().numpy()

    # Calculate accuracy
    accuracy = np.mean(predictions == y_test)

    print(f'Test Accuracy: {accuracy:.4f}')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Print the device being used
    print('Using device:', device)
    X = np.load(r"DataPreparation\x_data_all_withoutc4hard.npy")
    y = np.load(r"DataPreparation\y_data_all_withoutc4hard.npy")
    # X = np.load("DataPreparation/x_data.npy")
    # y = np.load("DataPreparation/y_data.npy")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42)
    
    # comment Justin: ToDo: normalize data (need to do this here instead of in data preparation,
    # because need to calculate normalization values only on training set and the apply same values to test set)

    e, lr = train_val(X_train, y_train, k=5)  # k-fold cross validation
    print(
        f"Best score result is a combination of {e} epochs and learning rate of {lr}")
    # train the model with the whole dataset
    train(X_train, y_train, epochs=e, learning_rate=lr)
    evaluate(X_test, y_test)  # evaluate the trained model
