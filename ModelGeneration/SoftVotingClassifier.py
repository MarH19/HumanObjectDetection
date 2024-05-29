import torch


class SoftVotingClassifier:
    def __init__(self, model, window_size=10):
        self.model = model
        self.window_size = window_size
        self.probability_window = []

    def update_probabilities(self, new_probabilities):
        self.probability_window.append(new_probabilities)
        if len(self.probability_window) > self.window_size:
            self.probability_window.pop(0)

    def soft_voting(self):
        if not self.probability_window or len(self.probability_window) < self.window_size:
            return None
        average_probabilities = torch.mean(
            torch.stack(self.probability_window), dim=0)
        final_prediction = torch.argmax(average_probabilities).item()
        return final_prediction

    def predict(self, data):
        # don't append data and predict if window is  full
        if len(self.probability_window) >= self.window_size:
            return None

        with torch.no_grad():
            output = self.model(data)
            probabilities = torch.nn.functional.softmax(
                output, dim=1).detach().cpu()
        self.update_probabilities(probabilities)

        # don't predict if window is not full
        if len(self.probability_window) < self.window_size:
            return None
        return self.soft_voting()

    def reset(self):
        self.probability_window = []
