import logging
import sys
from collections import OrderedDict

import torch
import torchvision.models as models
from torch.utils import model_zoo
from torchvision.models.resnet import BasicBlock, model_urls, Bottleneck

import skeleton
import torch.nn as nn
import math
import pdb

formatter = logging.Formatter(fmt='[%(asctime)s %(levelname)s %(filename)s] %(message)s')

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(handler)


class ResNet34(models.ResNet):
    Block = BasicBlock

    def __init__(self, in_channels, num_classes=10, **kwargs):
        Block = BasicBlock
        super(ResNet34, self).__init__(Block, [2, 2, 2, 2], num_classes=num_classes, **kwargs)  # resnet34

        if in_channels == 3:
            self.stem = torch.nn.Sequential(
                # skeleton.nn.Permute(0, 3, 1, 2),
                skeleton.nn.Normalize(0.5, 0.25, inplace=False),
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

        self.last_channels = 512 * Block.expansion
        self.conv1d = torch.nn.Sequential(
            skeleton.nn.Split(OrderedDict([
                ('skip', torch.nn.Sequential(
                )),
                ('deep', torch.nn.Sequential(
                    torch.nn.Conv1d(self.last_channels, self.last_channels,
                                    kernel_size=5, stride=1, padding=2, bias=False),
                    torch.nn.BatchNorm1d(self.last_channels),
                    torch.nn.ReLU(inplace=True),
                ))
            ])),
            skeleton.nn.MergeSum(),
            torch.nn.AdaptiveAvgPool1d(1)
        )

        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(self.last_channels, num_classes, bias=False)
        self._half = False
        self._class_normalize = True
        self._is_video = False

    def set_video(self, is_video=True, times=False):
        self._is_video = is_video
        if is_video:
            self.conv1d_prev = torch.nn.Sequential(
                skeleton.nn.SplitTime(times),
                skeleton.nn.Permute(0, 2, 1, 3, 4), # bs, c, t, h, w
            )

            self.conv1d_post = torch.nn.Sequential(
            )

    def is_video(self):
        return self._is_video

    def init(self, model_dir=None, gain=1.):
        self.model_dir = model_dir if model_dir is not None else self.model_dir
        sd = model_zoo.load_url(model_urls['resnet34'], model_dir=self.model_dir)
        # sd = model_zoo.load_url(model_urls['resnet34'], model_dir='./models/')
        del sd['fc.weight']
        del sd['fc.bias']
        self.load_state_dict(sd, strict=False)
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_parameters()
                
        for m in self.layer4.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc.weight, gain=gain)
        LOGGER.debug('initialize classifier weight')

    def forward_origin(self, x):
        x = self.conv1(x)
        # print('conv1', x.size())
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # print('layer1', x.size())
        x = self.layer2(x)
        # print('layer2', x.size())
        x = self.layer3(x)
        # print('layer3', x.size())
        x = self.layer4(x)
        # print('layer4', x.size())
        x = self.pool(x)
        # print('pool', x.size())

        if self.is_video():
            x = self.conv1d_prev(x) # bs, c, t, h , w
            x = x.view(x.size(0), x.size(1), -1) # bs, c, txhxw
            # print('view',x.shape)
            x = self.conv1d(x)
            x = self.conv1d_post(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # pdb.set_trace()
        return x

    def forward(self, inputs, targets=None, tau=8.0, reduction='avg'):  # pylint: disable=arguments-differ
        dims = len(inputs.shape) 

        if self.is_video() and dims == 5:
            batch, times, channels, height, width = inputs.shape
            inputs = inputs.view(batch * times, channels, height, width)

        # print('view', inputs.shape)

        inputs = self.stem(inputs)
        # print('stem', inputs.shape)
        logits = self.forward_origin(inputs)
        # print('forward', logits.shape)
        logits /= tau

        if targets is None:
            return logits
        if targets.device != logits.device:
            targets = targets.to(device=logits.device)

        loss = self.loss_fn(input=logits, target=targets)

        if self._class_normalize and isinstance(self.loss_fn, (
        torch.nn.BCEWithLogitsLoss, skeleton.nn.BinaryCrossEntropyLabelSmooth)):
            pos = (targets == 1).to(logits.dtype)
            neg = (targets < 1).to(logits.dtype)
            npos = pos.sum()
            nneg = neg.sum()

            positive_ratio = max(0.1, min(0.9, (npos) / (npos + nneg)))
            negative_ratio = max(0.1, min(0.9, (nneg) / (npos + nneg)))
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
