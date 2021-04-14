from automl_workflow.task import my_task
from automl_workflow.learner import MyLearner

import torch
import torchvision
import torchvision.transforms as transforms # Data Augmentor
from torch.utils.data import Dataset, DataLoader # Data Loader
import torch.nn as nn


class Evaluator():

    def __init__(self, task, learner):
        self.task = task
        self.learner = learner

    def train(self):
        train_set = self.task.train_set
        self.predictor = self.learner.learn(train_set)

    def test(self):
        test_set = self.task.test_set
        testloader = self.learner.data_loader(test_set, train=False)
        correct = 0
        total = 0
        metric = self.task.metric # revised using metric 
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # outputs = self.learner.backbone_model(images) # TODO: use predictor
                # _, predicted = torch.max(outputs.data, 1)
                predicted = self.predictor.predict(images) # revised using predictor
                
                score += metric(predicted, labels) # revised, 
                # TODO: clarify how to define the metric of each batch

                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()

        # print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
        # Revised, April 14, using metric


    def evaluate(self):
        self.train()
        self.test()


if __name__ == '__main__':
    task = my_task
    learner = MyLearner()
    evaluator = Evaluator(task, learner)
    evaluator.evaluate()

