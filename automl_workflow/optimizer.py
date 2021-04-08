import torch.optim as optim

import torch.nn as nn
from torch.optim import Adam

class MyOptimizer(object):

    def optimize(self, backbone_model, trainloader):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(backbone_model.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(2):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = backbone_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0