import torch.optim as optim

from torch.optim import Adam

class MyOptimizer(optim.Optimizer):

    def __init__(self):
        pass

MyOptimizer = Adam