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
        self.fc = nn.Linear(hidden_size, output_size)
        self.rnn_model = None

    def forward(self, x):
        pass

    def get_probabilities(self, logits):
        calc = nn.Sigmoid() if self.output_size == 1 else nn.Softmax(dim=1)
        return calc(logits)

    def get_predictions(self, logits):
        probabilities = self.get_probabilities(logits)
        if self.output_size == 1:
            return (probabilities.squeeze() > 0.5).cpu().numpy()
        else:
            return torch.argmax(probabilities, dim=1).cpu().numpy()

    def get_criterion(self):
        return nn.BCEWithLogitsLoss() if self.output_size == 1 else nn.CrossEntropyLoss()


class LSTMModel(RNNModel):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__(input_size=input_size,
                                        hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
        self.rnn_model = nn.LSTM(input_size, hidden_size,
                                 num_layers, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        lstm_out, _ = self.rnn_model(x, (h0, c0))
        return self.fc(lstm_out[:, -1, :])


class LSTMModelWithLayerNorm(LSTMModel):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModelWithLayerNorm, self).__init__(input_size=input_size,
                                        hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        out, _ = self.rnn_model(x, (h0, c0))
        out = self.layer_norm(out)
        return self.fc(out[:, -1, :])


class GRUModel(RNNModel):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__(input_size=input_size,
                                       hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
        self.rnn_model = nn.GRU(input_size, hidden_size,
                                num_layers, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        gru_out, _ = self.rnn_model(x, h0)
        return self.fc(gru_out[:, -1, :])
    
    
class GRUModelWithLayerNorm(GRUModel):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModelWithLayerNorm, self).__init__(input_size=input_size,
                                                    hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        out, _ = self.rnn_model(x, h0)
        out = self.layer_norm(out)
        return self.fc(out[:, -1, :])
