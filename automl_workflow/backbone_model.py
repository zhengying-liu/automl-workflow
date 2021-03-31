# from automl_workflow.api import BaseDataTransformer
# TODO: implement linear model (no hidden layer), ResNet18 (can be copied from DeepWisdom or directly from PyTorch)

import torch.nn as nn

class MyBackboneModel1(nn.Module):
    """Linear model."""
    
    def __init__(self):
        pass

    def forward(self, x):
        return x


class MyBackboneModel2(nn.Module):
    """ResNet18."""
    
    def __init__(self):
        pass

    def forward(self, x):
        return x


MyBackboneModel = MyBackboneModel1