import logging
import sys
from collections import OrderedDict

import torch
import torchvision.models.video.resnet as models
from torch.utils import model_zoo
from torchvision.models.video.resnet import BasicBlock, model_urls, Conv3DSimple, Conv3DNoTemporal, BasicStem, \
    R2Plus1dStem, Conv2Plus1D
from itertools import chain

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


class ResNet(models.VideoResNet):
    def __init__(self, in_channels, num_classes=10, **kwargs):
        Block = BasicBlock
        super(ResNet, self).__init__(Block, num_classes=num_classes,
                                     conv_makers=[Conv3DSimple] + [Conv3DNoTemporal] * 3, layers=[2, 2, 2, 2],
                                     stem=BasicStem, **kwargs)
        # super(ResNet, self).__init__(Block, num_classes=num_classes, conv_makers=[Conv2Plus1D] * 4, layers=[2, 2, 2, 2], stem=R2Plus1dStem, **kwargs) 

        self._class_normalize = True
        self._is_video = True
        self._half = False
        self.init_hyper_params()
        self.checkpoints = []
        self.predict_prob_list =dict()
        self.round_idx = 0
        self.single_ensemble = False
        self.use_test_time_augmentation = False
        self.update_transforms = False
        self.history_predictions = dict()
        self.g_his_eval_dict = dict()
        self.last_y_pred_round = 0

        if in_channels == 3:
            self.preprocess = torch.nn.Sequential(
                # skeleton.nn.Permute(0, 3, 1, 2),
                # skeleton.nn.Permute(0, 2, 1, 3, 4),
                skeleton.nn.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989], mode='conv3d',inplace=False),
            )
        elif in_channels == 1:
            self.preprocess = torch.nn.Sequential(
                # skeleton.nn.Permute(0, 2, 1, 3,4),
                skeleton.nn.Normalize(0.5, 0.25, mode='conv3d',inplace=False),
                skeleton.nn.CopyChannels(3),
            )
        else:
            self.preprocess = torch.nn.Sequential(
                # skeleton.nn.Permute(0, 3, 1, 2),
                skeleton.nn.Normalize(0.5, 0.25,mode='conv3d', inplace=False),
                torch.nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(3),
            )

        self.last_channels = 512 * Block.expansion
        self.fc = torch.nn.Linear(self.last_channels, num_classes, bias=False)

    def init_hyper_params(self):
        self.info = {
            'loop': {
                'epoch': 0,
                'test': 0,
                'best_score': 0.0
            },
            'condition': {
                'first': {
                    'train': True,
                    'valid': True,
                    'test': True
                }
            },
            'terminate': False
        }

        # TODO: adaptive logic for hyper parameter
        self.hyper_params = {
            'optimizer': {
                'lr': 0.025,
                'warmup_multiplier':2.0,
                'warmup_epoch':3
            },
            'dataset': {
                'train_info_sample': 256,
                'cv_valid_ratio': 0.1,
                'max_valid_count': 256,

                'max_size': 64,
                'base': 16,  # input size should be multipliers of 16
                'max_times': 8,

                'enough_count': {
                    'image': 10000,
                    'video': 1000
                },

                'batch_size': 32,
                'steps_per_epoch': 30,
                'max_epoch': 1000,  # initial value
                'batch_size_test': 256,
            },
            'checkpoints': {
                'keep': 50
            },
            'conditions': {
                'score_type': 'auc',
                'early_epoch': 1,
                'skip_valid_score_threshold': 0.90,  # if bigger then 1.0 is not use
                'test_after_at_least_seconds': 1,
                'test_after_at_least_seconds_max': 90,
                'test_after_at_least_seconds_step': 2,

                'threshold_valid_score_diff': 0.001,
                'threshold_valid_best_score': 0.997,
                'decide_threshold_valid_best_score': 0.9300,
                'max_inner_loop_ratio': 0.1,
                'min_lr': 1e-6,
                'use_fast_auto_aug': True
            }
        }

    def init(self, model_dir=None, gain=1.):
        self.model_dir = model_dir if model_dir is not None else self.model_dir
        sd = model_zoo.load_url(model_urls['mc3_18'], model_dir=self.model_dir)
        del sd['fc.weight']
        del sd['fc.bias']
        # for m in self.layer1.modules():
        #     for p in m.parameters():
        #         p.requires_grad_(False)
        # for m in self.stem.modules():
        #     for p in m.parameters():
        #         p.requires_grad_(False)
        self.load_state_dict(sd, strict=False)
        torch.nn.init.xavier_uniform_(self.fc.weight, gain=gain)
        LOGGER.debug('initialize classifier weight')

    def init_opt(self,steps_per_epoch,batch_size,init_lr,warmup_multiplier,warm_up_epoch):
        lr_multiplier = max(0.5, batch_size / 32)
        
        params = [p for p in self.parameters() if p.requires_grad]
        params_fc = [p for n, p in self.named_parameters() if
                     p.requires_grad and 'fc' == n[:2] or 'conv1d' == n[:6]]
        
        scheduler_lr = skeleton.optim.get_change_scale(
            skeleton.optim.gradual_warm_up(
                skeleton.optim.get_reduce_on_plateau_scheduler(
                    init_lr * lr_multiplier / warmup_multiplier,
                    patience=10, factor=.5, metric_name='train_loss'
                ),
                warm_up_epoch=warm_up_epoch,
                multiplier=warmup_multiplier
            ),
            init_scale=1.0
        )

        # scheduler_fc_lr = skeleton.optim.get_change_scale(
        #     skeleton.optim.gradual_warm_up(
        #         skeleton.optim.get_reduce_on_plateau_scheduler(
        #             init_lr * 10 * lr_multiplier / warmup_multiplier,
        #             patience=10, factor=.5, metric_name='train_loss'
        #         ),
        #         warm_up_epoch=warm_up_epoch,
        #         multiplier=warmup_multiplier
        #     ),
        #     init_scale=1.0
        # )

        self.optimizer_fc = skeleton.optim.ScheduledOptimizer(
            params_fc,
            torch.optim.SGD,
            steps_per_epoch=steps_per_epoch,
            clip_grad_max_norm=None,
            lr=scheduler_lr,
            momentum=0.9,
            weight_decay=0.00025,
            nesterov=True
        )
        self.optimizer = skeleton.optim.ScheduledOptimizer(
            params,
            torch.optim.SGD,
            steps_per_epoch=steps_per_epoch,
            clip_grad_max_norm=None,
            lr=scheduler_lr,
            momentum=0.9,
            weight_decay=0.00025,
            nesterov=True
        )

    def set_video(self, is_video=True, times=False):
        self._is_video = is_video
        if is_video:
            self.conv1d_prev = torch.nn.Sequential(
                skeleton.nn.SplitTime(times),
                skeleton.nn.Permute(0, 2, 1, 3, 4),  # bs, c, t, h, w
            )

            self.conv1d_post = torch.nn.Sequential(
            )

    def forward_origin(self, x):
        # print(x.shape)
        x = self.preprocess(x)
        # print(x.shape)

        x = self.stem(x)
        # print('stem', x.size())
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print('layer4', x.size())
        x = self.avgpool(x)
        # Flatten the layer to fc
        x = x.flatten(1)
        x = self.fc(x)

        return x

    def forward(self, inputs, targets=None, tau=8.0, reduction='avg'):
        dims = len(inputs.shape)
        # if self.is_video() and dims == 5:
        # batch, times, channels, height, width = inputs.shape
        # inputs = inputs.view(batch * times, channels, height, width)

        # print('===> is_video',self.is_video())

        # inputs = self.stem(inputs)
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


        #     # positive_ratio = max(0.1, min(0.9, (npos) / (npos + nneg)))
        #     positive_ratio = torch.clamp((npos) / (npos + nneg), min=0.1, max=0.9).view(1,loss.shape[1])
        #     # print(positive_ratio)
        #     # negative_ratio = max(0.1, min(0.9, (nneg) / (npos + nneg)))
        #     negative_ratio = torch.clamp((nneg) / (npos + nneg), min=0.1, max=0.9).view(1,loss.shape[1])
        #     # print(negative_ratio)
        #     LOGGER.debug('[BCEWithLogitsLoss] positive_ratio:%f, negative_ratio:%f',
        #                 positive_ratio, negative_ratio)

        #     normalized_loss = (loss * pos) / positive_ratio
        #     normalized_loss += (loss * neg) / negative_ratio

        #     loss = normalized_loss

        # if reduction == 'avg':
        #     loss = loss.mean()
        # elif reduction == 'max':
        #     loss = loss.max()
        # elif reduction == 'min':
        #     loss = loss.min()
        # # print(f'Logits shape {logits.shape}')
        # # pdb.set_trace()
        # return logits, loss