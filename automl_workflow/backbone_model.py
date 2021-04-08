# from automl_workflow.api import BaseDataTransformer
# TODO: implement linear model (no hidden layer), ResNet18 (can be copied from DeepWisdom or directly from PyTorch)

import torch.nn as nn
import torch.nn.functional as F

# inplementation of a linear model
class MyBackboneModel1(nn.Module):
    """Linear model."""
    
    def __init__(self):
        super(MyBackboneModel1, self).__init__()
        self.fc = nn.Linear(4 * 3 * 32 * 32, 1)

    def forward(self, x):
        x = x.view(-1)
        x = self.fc(x)
        return x


# inplementation of ResNet
# This can be done in one line in pytorch, so I didn't include the code here 
class MyBackboneModel2(nn.Module):
    """ResNet18."""
    
    def __init__(self):
        pass

    def forward(self, x):
        return x

class MyBackboneModel3(nn.Module):
    """A custom CNN in PyTorch tutorial."""
    
    def __init__(self):
        super(MyBackboneModel3, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


MyBackboneModel = MyBackboneModel3