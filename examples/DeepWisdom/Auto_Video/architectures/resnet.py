import logging
import sys
from collections import OrderedDict

import torch
import torchvision.models as models
from torch.utils import model_zoo
from torchvision.models.resnet import BasicBlock, model_urls, Bottleneck
import pdb
import skeleton

formatter = logging.Formatter(fmt='[%(asctime)s %(levelname)s %(filename)s] %(message)s')

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)

LOGGER = logging.getLogger(__name__) 
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(handler)


class ResNet18(models.ResNet):
    Block = BasicBlock

    def __init__(self, in_channels, num_classes=10, **kwargs):
        Block = BasicBlock
        super(ResNet18, self).__init__(Block, [2, 2, 2, 2], num_classes=num_classes, **kwargs)    # resnet18

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
        self.ensemble_scores =dict()
        self.ensemble_predictions = dict()
        self.ensemble_test_index = 0
        
        if in_channels == 3:
            self.stem = torch.nn.Sequential(
                # skeleton.nn.Permute(0, 3, 1, 2),
                skeleton.nn.Normalize(0.5, 0.25, mode='conv2d',inplace=False),
            )
        elif in_channels == 1:
            self.stem = torch.nn.Sequential(
                # skeleton.nn.Permute(0, 3, 1, 2),
                skeleton.nn.Normalize(0.5, 0.25, mode='conv2d',inplace=False),
                skeleton.nn.CopyChannels(3),
            )
        else:
            self.stem = torch.nn.Sequential(
                # skeleton.nn.Permute(0, 3, 1, 2),
                skeleton.nn.Normalize(0.5, 0.25,mode='conv2d', inplace=False),
                torch.nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(3),
            )

        self.last_channels = 512 * Block.expansion
        self.conv1d = torch.nn.Sequential(
            skeleton.nn.Split(OrderedDict([
                ('skip', torch.nn.Sequential(
                    # torch.nn.AvgPool1d(3, stride=2, padding=1)
                )),
                ('deep', torch.nn.Sequential(
                    # torch.nn.Conv1d(self.last_channels, self.last_channels // 4,
                    #                 kernel_size=1, stride=1, padding=0, bias=False),
                    # torch.nn.BatchNorm1d(self.last_channels // 4),
                    # torch.nn.ReLU(inplace=True),
                    # torch.nn.Conv1d(self.last_channels // 4, self.last_channels // 4,
                    #                 kernel_size=5, stride=1, padding=2, groups=self.last_channels // 4, bias=False),
                    # torch.nn.BatchNorm1d(self.last_channels // 4)

                    # torch.nn.ReLU(inplace=True),
                    # torch.nn.Conv1d(self.last_channels // 4, self.last_channels,
                    #                 kernel_size=1, stride=1, padding=0, bias=False),
                    # torch.nn.BatchNorm1d(self.last_channels),

                    torch.nn.Conv1d(self.last_channels, self.last_channels,
                                    kernel_size=5, stride=1, padding=2, bias=False),
                    torch.nn.BatchNorm1d(self.last_channels),
                    torch.nn.ReLU(inplace=True),
                ))
            ])),
            skeleton.nn.MergeSum(),

            # torch.nn.Conv1d(self.last_channels, self.last_channels,
            #                 kernel_size=5, stride=1, padding=2, bias=False),
            # torch.nn.BatchNorm1d(self.last_channels),
            # torch.nn.ReLU(inplace=True),

            torch.nn.AdaptiveAvgPool1d(1)
        )

        self.permute =  skeleton.nn.Permute(0, 2, 1, 3, 4)
        # self.self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        # self.pool = torch.nn.AdaptiveMaxPool2d((1, 1))
        self.fc = torch.nn.Linear(self.last_channels, num_classes, bias=False)
        self._half = False
        self._class_normalize = True
        self._is_video = False

    def set_video(self, is_video=True, times=False):
        self._is_video = is_video
        if is_video:
            self.conv1d_prev = torch.nn.Sequential(
                skeleton.nn.SplitTime(times),
                skeleton.nn.Permute(0, 2, 1, 3, 4),
            )

            self.conv1d_post = torch.nn.Sequential(
            )

    def is_video(self):
        return self._is_video

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
                'warmup_epoch':5
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

    def init(self, model_dir=None, gain=1.):
        self.model_dir = model_dir if model_dir is not None else self.model_dir
        sd = model_zoo.load_url(model_urls['resnet18'], model_dir=self.model_dir)
        # sd = model_zoo.load_url(model_urls['resnet34'], model_dir='./models/')
        del sd['fc.weight']
        del sd['fc.bias']
        self.load_state_dict(sd, strict=False)

        # for idx in range(len(self.stem)):
        #     m = self.stem[idx]
        #     if hasattr(m, 'weight') and not isinstance(m, torch.nn.BatchNorm2d):
        #         # torch.nn.init.kaiming_normal_(self.stem.weight, mode='fan_in', nonlinearity='linear')
        #         torch.nn.init.xavier_normal_(m.weight, gain=gain)
        #         LOGGER.debug('initialize stem weight')
        #
        # for idx in range(len(self.conv1d)):
        #     m = self.conv1d[idx]
        #     if hasattr(m, 'weight') and not isinstance(m, torch.nn.BatchNorm1d):
        #         # torch.nn.init.kaiming_normal_(self.stem.weight, mode='fan_in', nonlinearity='linear')
        #         torch.nn.init.xavier_normal_(m.weight, gain=gain)
        #         LOGGER.debug('initialize conv1d weight')

        # torch.nn.init.kaiming_uniform_(self.fc.weight, mode='fan_in', nonlinearity='sigmoid')
        
        torch.nn.init.xavier_uniform_(self.fc.weight, gain=gain)
        LOGGER.debug('initialize classifier weight')

    def forward_origin(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)

        if self.is_video():
            x = self.conv1d_prev(x)
            x = x.reshape(x.size(0), x.size(1), -1)
            x = self.conv1d(x)
            x = self.conv1d_post(x)

        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, inputs, targets=None, tau=8.0, reduction='avg'):  # pylint: disable=arguments-differ
        dims = len(inputs.shape)
        # pdb.set_trace()
        if self.is_video() and dims == 5:
            inputs = self.permute(inputs)
            batch, times, channels, height, width = inputs.shape
            inputs = inputs.reshape(batch*times, channels, height, width)
        # import pdb; pdb.set_trace()
        inputs = self.stem(inputs)
        logits = self.forward_origin(inputs)
        logits /= tau

        if targets is None:
            return logits
        if targets.device != logits.device:
            targets = targets.to(device=logits.device)

        loss = self.loss_fn(input=logits, target=targets)

        if self._class_normalize and isinstance(self.loss_fn, (torch.nn.BCEWithLogitsLoss, skeleton.nn.BinaryCrossEntropyLabelSmooth)):
            pos = (targets == 1).to(logits.dtype)
            neg = (targets < 1).to(logits.dtype)
            npos = pos.sum()
            nneg = neg.sum()

            positive_ratio = max(0.1, min(0.9, (npos) / (npos + nneg)))
            negative_ratio = max(0.1, min(0.9, (nneg) / (npos + nneg)))
            LOGGER.debug('[BCEWithLogitsLoss] positive_ratio:%f, negative_ratio:%f',
                         positive_ratio, negative_ratio)

            normalized_loss =  (loss * pos) / positive_ratio
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
        # super(BasicNet, self).half()
        for module in self.modules():
            if len([c for c in module.children()]) > 0:
                continue

            if not isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                module.half()
            else:
                module.float()
        self._half = True
        return self
