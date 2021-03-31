# from automl_workflow.api import DataLoader

from torch.utils.data import Dataset, DataLoader

class MyDataLoader(DataLoader):
    """TODO: default PyTorch dataloader for train, validation, test, etc."""
    
    def __init__(self, mode='train', *argv, **kwargs):
        pass