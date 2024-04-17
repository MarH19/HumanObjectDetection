import torch
import torch.nn as nn


class RNNModelHyperParameterSet():
    def __init__(self, hidden_size: int = None, num_layers: int = None, epochs: int = None, learning_rate: float = None):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.learning_rate = learning_rate


class RNNModelHyperParameters():
    def __init__(self, hidden_sizes: list[int], num_layers: list[int], epochs: list[int], learning_rates: list[float], input_size: int, output_size: int):
        self.hidden_sizes = hidden_sizes
        self.num_layers = num_layers
        self.epochs = epochs
        self.learning_rates = learning_rates

        self.input_size = input_size
        self.output_size = output_size

        self.best_hyperparameters = RNNModelHyperParameterSet()

    def get_hyperparameter_combinations(self) -> list[RNNModelHyperParameterSet]:
        return [RNNModelHyperParameterSet(h, n, e, l) for h in self.hidden_sizes for n in self.num_layers for e in self.epochs for l in self.learning_rates]


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

    def forward(self, x):
        pass


class LSTMModel(RNNModel):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__(input_size=input_size,
                                        hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.output_calc = nn.Sigmoid() if output_size == 1 else nn.Softmax(dim=1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        z = self.fc(lstm_out[:, -1, :])
        return self.output_calc(z)
