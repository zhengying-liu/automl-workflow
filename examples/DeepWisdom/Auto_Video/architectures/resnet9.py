import logging
import sys
from collections import OrderedDict
import copy
import torch
import torchvision.models as models
from torch.utils import model_zoo
# from torchvision.models.resnet import BasicBlock, model_urls, Bottleneck
from torchvision.models.resnet import model_urls, Bottleneck
import skeleton
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy.special import binom
formatter = logging.Formatter(fmt='[%(asctime)s %(levelname)s %(filename)s] %(message)s')
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(handler)

class ResNet9(nn.Module):
    # Block = BasicBlock
    def __init__(self, in_channels, num_classes=10, **kwargs):
        # Block = BasicBlock
        super(ResNet9, self).__init__()  # resnet18
        if in_channels == 3:
            self.stem = torch.nn.Sequential(
                # skeleton.nn.Permute(0, 3, 1, 2),
                skeleton.nn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=False),
            )
        elif in_channels == 1:
            self.stem = torch.nn.Sequential(
                # skeleton.nn.Permute(0, 3, 1, 2),
                skeleton.nn.Normalize(0.5, 0.25, inplace=False),
                skeleton.nn.CopyChannels(3),
            )
        else:
            self.stem = torch.nn.Sequential(
                # skeleton.nn.Permute(0, 3, 1, 2),
                skeleton.nn.Normalize(0.5, 0.25, inplace=False),
                torch.nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(3),
            )

        self.premute = skeleton.nn.Permute(0, 2 ,1, 3, 4)

        conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                          bias=False)
        norm1 = nn.BatchNorm2d(num_features=64)
        act = nn.CELU(0.075, inplace=False)
        pool = nn.MaxPool2d(2)
        self.prep = nn.Sequential(OrderedDict([('conv', conv1), ('bn', norm1), ('act', act)]))
        self.layer1 = ResLayer(64, 128)
        conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                          bias=False)
        norm2 = nn.BatchNorm2d(num_features=256)
        self.layer2 = nn.Sequential(OrderedDict([('conv', conv2), ('pool', pool), ('bn', norm2), ('act', act)]))
        self.layer3 = ResLayer(256, 512)
        self.pool4 = nn.AdaptiveMaxPool2d(1)
        # self.conv3 = nn.Linear(512, num_classes, bias=False)
        self.fc = torch.nn.Linear(512, num_classes, bias=False)
        self._half = False
        self._class_normalize = True

    def init(self, model_dir=None, gain=1.):
        self.model_dir = model_dir if model_dir is not None else self.model_dir
        # sd = model_zoo.load_url(model_urls['resnet18'], model_dir=self.model_dir)
        # sd = model_zoo.load_url(model_urls['resnet34'], model_dir='./models/')
        # self.model_dir = model_dir if model_dir is not None else self.model_dir
        sd = torch.load(self.model_dir + '/checkpoint.pth.tar')
        # print(sd)
        # print(sd['state_dict'])
        new_sd = copy.deepcopy(sd['state_dict'])
        for key,value in sd['state_dict'].items():
            new_sd[key[7:]] = sd['state_dict'][key]
        # del new_sd['fc.weight']
        # del new_sd['fc.bias']
        self.load_state_dict(new_sd, strict=False)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc.weight, gain=gain)

        LOGGER.debug('initialize classifier weight')
    def reset_255(self):
        if self.in_channels == 3:
            self.stem = torch.nn.Sequential(
                # skeleton.nn.Permute(0, 3, 1, 2),
                skeleton.nn.Normalize([0.485, 0.456, 0.406] * 255, [0.229, 0.224, 0.225] * 255, inplace=False),
            ).cuda()
        elif self.in_channels == 1:
            self.stem = torch.nn.Sequential(
                # skeleton.nn.Permute(0, 3, 1, 2),
                skeleton.nn.Normalize(0.5 * 255, 0.25 * 255, inplace=False),
                skeleton.nn.CopyChannels(3),
            ).cuda()
        else:
            self.self.stem = torch.nn.Sequential(
                # skeleton.nn.Permute(0, 3, 1, 2),
                skeleton.nn.Normalize(0.5 * 255, 0.25 * 255, inplace=False),
                torch.nn.Conv2d(self.in_channels, 3, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(3),
            ).cuda()

    def forward_origin(self, x, targets):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool4(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, inputs, targets=None, tau=8.0, reduction='avg'):  # pylint: disable=arguments-differ
        
        dims = len(inputs.shape)

        if self.is_video() and dims == 5:
            inputs = self.premute(inputs)
            batch, times, channels, height, width = inputs.shape
            inputs = inputs.view(batch*times, channels, height, width)
        
        inputs = self.stem(inputs)
        logits = self.forward_origin(inputs, targets)
        logits /= tau
        if targets is None:
            return logits
        if targets.device != logits.device:
            targets = targets.to(device=logits.device)
        loss = self.loss_fn(input=logits, target=targets)
        
        if self._class_normalize and isinstance(self.loss_fn, (torch.nn.BCEWithLogitsLoss, skeleton.nn.BinaryCrossEntropyLabelSmooth)):
            pos = (targets == 1).to(logits.dtype)
            neg = (targets < 1).to(logits.dtype)
            npos = pos.sum(dim=0)
            nneg = neg.sum(dim=0)
            positive_ratio = torch.clamp((npos) / (npos + nneg), min=0.1, max=0.9).view(1, loss.shape[1])
            negative_ratio = torch.clamp((nneg) / (npos + nneg), min=0.1, max=0.9).view(1, loss.shape[1])
            LOGGER.debug('[BCEWithLogitsLoss] positive_ratio:%f, negative_ratio:%f',
                         positive_ratio, negative_ratio)
            normalized_loss = (loss * pos) / positive_ratio
            normalized_loss += (loss * neg) / negative_ratio
            loss = normalized_loss

        if reduction == 'avg':
            loss = loss.mean()
        elif reduction == 'max':
            loss = loss.max()
        elif reduction == 'min':
            loss = loss.min()
        return logits, loss

    def half(self):
        for module in self.modules():
            if len([c for c in module.children()]) > 0:
                continue
            if not isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                module.half()
            else:
                module.float()
        self._half = True
        return self