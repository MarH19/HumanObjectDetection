import torch
import torch.nn as nn


class RNNModelHyperParameterSet():
    def __init__(self, hidden_size: int = None, num_layers: int = None, epochs: int = None, learning_rate: float = None, dropout_rate: float = None):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate


class RNNModelHyperParameters():
    def __init__(self, hidden_sizes: "list[int]", num_layers: "list[int]", epochs: "list[int]", learning_rates: "list[float]", dropout_rates: "list[float]", input_size: int, output_size: int):
        self.hidden_sizes = hidden_sizes
        self.num_layers = num_layers
        self.epochs = epochs
        self.learning_rates = learning_rates
        self.dropout_rates = dropout_rates

        self.input_size = input_size
        self.output_size = output_size

        self.best_hyperparameters = RNNModelHyperParameterSet()

    def get_hyperparameter_combinations(self) -> "list[RNNModelHyperParameterSet]":
        return [RNNModelHyperParameterSet(h, n, e, l, d) for h in self.hidden_sizes for n in self.num_layers for e in self.epochs for l in self.learning_rates for d in (self.dropout_rates if self.dropout_rates is not None else [None])]


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=None, layer_norm=None):
        super(RNNModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.fc = nn.Linear(hidden_size, output_size)
        self.layer_norm = layer_norm
        self.dropout = nn.Dropout(
            dropout_rate) if dropout_rate is not None else None
        self.rnn_layers = nn.ModuleList()

    def forward(self, x):
        for _, rnn_layer in enumerate(self.rnn_layers):
            x, _ = rnn_layer(x, self.get_bias(x))
            x = self.layer_norm(x) if self.layer_norm is not None else x
            x = self.dropout(x) if self.dropout is not None else x
        return self.fc(x[:, -1, :])

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

    def get_bias(self):
        pass


class LSTMModel(RNNModel):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=None, layer_norm=None):
        super(LSTMModel, self).__init__(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                        output_size=output_size, dropout_rate=dropout_rate, layer_norm=layer_norm)
        for l in range(num_layers):
            self.rnn_layers.append(
                nn.LSTM(input_size if l == 0 else hidden_size, hidden_size, batch_first=True))

    def get_bias(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        return (h0, c0)


class LSTMModelWithLayerNorm(LSTMModel):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=None):
        super(LSTMModelWithLayerNorm, self).__init__(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                                     output_size=output_size, dropout_rate=dropout_rate, layer_norm=nn.LayerNorm(hidden_size))


class GRUModel(RNNModel):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=None, layer_norm=None):
        super(GRUModel, self).__init__(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                       output_size=output_size, dropout_rate=dropout_rate, layer_norm=layer_norm)
        for l in range(num_layers):
            self.rnn_layers.append(nn.GRU(
                input_size if l == 0 else hidden_size, hidden_size, batch_first=True))

    def get_bias(self, x):
        return (torch.zeros(1, x.size(0), self.hidden_size).to(x.device))


class GRUModelWithLayerNorm(GRUModel):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=None):
        super(GRUModelWithLayerNorm, self).__init__(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                                    output_size=output_size, dropout_rate=dropout_rate, layer_norm=nn.LayerNorm(hidden_size))
