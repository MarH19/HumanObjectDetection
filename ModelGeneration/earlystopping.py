import torch


class EarlyStopper:
    def __init__(self, max_epochs, patience=10, min_delta=0.1):
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss, model, model_params_path, epoch):
        #if epoch < int((2 / 3) * self.max_epochs):
        #    return False
        
        if validation_loss < self.min_validation_loss:
            self.counter = 0
            self.min_validation_loss = validation_loss
            torch.save(model.state_dict(), model_params_path)
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
