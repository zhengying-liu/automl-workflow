# from automl_workflow.api import DataLoader

import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from skimage import io, transform
import numpy as np

class MyDataLoader(object):
    """TODO: default PyTorch dataloader for train, validation, test, etc."""
    
    def __call__(self, dataset, train=True):
        pt_dataloader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                    shuffle=train, num_workers=2)
        return pt_dataloader

# DEMO

'''
For the Dataset to be processed by the PyTorch Dataloader, we need to implement
the `__len__` method and the `__getitem__` method.
'''
class FaceDataSet(Dataset):
    """Face Landmarks dataset from pytorch tutorial."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    # must implement this method
    def __len__(self):
        return len(self.landmarks_frame)

    # must implement this method
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


'''
Custom implementation of ToTensor that process the output of the dataset. 
The process is necessary because we also include auxiliary data of the dataset
in this demo (landmark). 
'''
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    # major method to be implemented
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}

'''
Custom implementation of Rescale that process the output of the dataset. 
'''
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


def face_dataset_demo():
    # Demo for using this face dataset
    composed = transforms.Compose([Rescale((224, 224)), ToTensor()])
    transformed_dataset = FaceDataSet(csv_file='faces/face_landmarks.csv',
                                            root_dir='faces/',
                                            transform=composed)

    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]
        print(i, sample['image'].size(), sample['landmarks'].size())
        if i == 3:
            print('-' * 10)
            break

    dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=0)
    
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(), sample_batched['landmarks'].size())

# face_dataset_demo()


### Import CIFAR-10 training data loader ###

from automl_workflow.cifar10_data_loader import trainloader, testloader, trainset, testset

MyTrainSet = lambda: trainset
MyTestSet = lambda: testset
MyTrainDataLoader = lambda: trainloader
MyTestDataLoader = lambda: testloader