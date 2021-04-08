# Orginizers should implement the following classes
from automl_workflow.data_loader import MyTrainSet
from automl_workflow.data_loader import MyTestSet

from automl_workflow.api import Task, Metric

import torch


class MyMetric(Metric):

    def __call__(self, y_pred, y_true):
        """
        Args:
          y_pred: torch.Tensor of predicted labels
          y_true: torch.Tensor of true labels
        """
        total = y_true.size(0)
        correct = (y_pred == y_true).sum().item()
        accuracy = correct / total
        return accuracy


train_set = MyTrainSet()
test_set = MyTestSet()
metric = MyMetric()

my_task = Task(train_set, metric, test_set=test_set)