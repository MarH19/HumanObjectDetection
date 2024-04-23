import math
import torch
from torch.optim.optimizer import Optimizer


def get_optimizer(name):

    if name == "Adam":
        return torch.optim.Adam
    elif name == "RAdam":
        return torch.optim.RAdam


