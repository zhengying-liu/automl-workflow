import torch
import torchvision
import torchvision.transforms as transforms
import backbone_model
import torchvision.models as models


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

'''
If needed to download, change to download=True
'''
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=False)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


if __name__ == '__main__':
    # net = backbone_model.MyBackboneModel1()
    # net = models.resnet18(pretrained=True)
    net = backbone_model.MyBackboneModel3()
    print(net)

    for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            output = net(inputs)
            print(output)
            break