import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

def train(X_train,y_train,epochs,learning_rate):
    # Define model parameters
    input_size = 7  # number of features
    hidden_size = 64  # number of LSTM units
    num_layers = 1  # number of LSTM layers
    output_size = 1  # binary classification

    # Create LSTM model
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    model = model.to(device)
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train,dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train,dtype=torch.float32).unsqueeze(1).to(device) 
    print(X_train_tensor.shape)
    # Define loss function and optimizer
    #TODO change criterion when not binary anymore
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    torch.save(model.state_dict(), 'lstm_model1.pth')
def evaluate(X_test, y_test):
    # Load the trained model
    model = LSTMModel(input_size=7, hidden_size=64, num_layers=1, output_size=1)
    model.load_state_dict(torch.load('lstm_model1.pth'))
    model.eval()

    # Convert numpy arrays to PyTorch tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    print("test tensor:",X_test_tensor.shape)
    # Perform inference
    with torch.no_grad():
        outputs = model(X_test_tensor)
        predictions = (outputs.squeeze() > 0.5).cpu().numpy()  # Convert to binary predictions

    # Calculate accuracy
    accuracy = np.mean(predictions == y_test)

    print(f'Test Accuracy: {accuracy:.4f}')
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Print the device being used
    print('Using device:', device)
    #X = np.load("DataPreparation/x_data.npy")
    #y = np.load("DataPreparation/y_data.npy")
    X = np.load("/home/mindlab/humanObjectDetection/DataPreparation/x_data.npy")
    y = np.load("/home/mindlab/humanObjectDetection/DataPreparation/y_data.npy")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.1, random_state=42)
    train(X_train,y_train,epochs=100,learning_rate=0.001)
    evaluate(X_test, y_test)
