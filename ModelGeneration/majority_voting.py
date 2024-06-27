from abc import ABC, abstractmethod

import numpy as np
import torch


class MajorityVotingClassifier(ABC):

    def __init__(self, model, nof_individual_predictions=10):
        self.model = model
        self.nof_individual_predictions = nof_individual_predictions
        self.probability_window = []

    def update_probabilities(self, new_probabilities):
        self.probability_window.append(new_probabilities)
        if len(self.probability_window) > self.nof_individual_predictions:
            self.probability_window.pop(0)

    def majority_voting(self):
        if not self.probability_window or len(self.probability_window) < self.nof_individual_predictions:
            return None
        return self.do_majority_voting()

    def predict(self, data):
        # don't append data and predict if window is  full
        if len(self.probability_window) >= self.nof_individual_predictions:
            return None

        with torch.no_grad():
            output = self.model(data)
            probabilities = torch.nn.functional.softmax(
                output, dim=1).detach().cpu()
        self.update_probabilities(probabilities)

        # don't predict if window is not full
        if len(self.probability_window) < self.nof_individual_predictions:
            return None
        return self.majority_voting()

    def reset(self):
        self.probability_window = []

    def get_has_predicted(self):
        return len(self.probability_window) >= self.nof_individual_predictions

    @abstractmethod
    def do_majority_voting(self):
        pass


class SoftVotingClassifier(MajorityVotingClassifier):

    def __init__(self, model, nof_individual_predictions=10):
        super().__init__(model, nof_individual_predictions)

    def do_majority_voting(self):
        average_probabilities = torch.mean(
            torch.stack(self.probability_window), dim=0)
        final_prediction = torch.argmax(average_probabilities).item()
        return final_prediction


class HardVotingClassifier(MajorityVotingClassifier):

    def __init__(self, model, nof_individual_predictions=10):
        super().__init__(model, nof_individual_predictions)

    def do_majority_voting(self):
        predictions = [torch.argmax(probabilities, dim=1).item()
                       for probabilities in self.probability_window]
        final_prediction = np.bincount(predictions).argmax()
        return final_prediction
