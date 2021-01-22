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


class LSoftmaxLinear(nn.Module):

    def __init__(self, input_features, output_features, margin, device):
        super().__init__()
        self.input_dim = input_features  # number of input feature i.e. output of the last fc layer
        self.output_dim = output_features  # number of output = class numbers
        self.margin = margin  # m
        self.beta = 100
        self.beta_min = 0
        self.scale = 0.99

        self.device = device  # gpu or cpu

        # Initialize L-Softmax parameters
        self.weight = nn.Parameter(torch.FloatTensor(input_features, output_features))
        self.divisor = math.pi / self.margin  # pi/m
        self.C_m_2n = torch.Tensor(binom(margin, range(0, margin + 1, 2))).to(device)  # C_m{2n}
        self.cos_powers = torch.Tensor(range(self.margin, -1, -2)).to(device)  # m - 2n
        self.sin2_powers = torch.Tensor(range(len(self.cos_powers))).to(device)  # n
        self.signs = torch.ones(margin // 2 + 1).to(device)  # 1, -1, 1, -1, ...
        self.signs[1::2] = -1

    def calculate_cos_m_theta(self, cos_theta):
        sin2_theta = 1 - cos_theta ** 2
        cos_terms = cos_theta.unsqueeze(1) ** self.cos_powers.unsqueeze(0)  # cos^{m - 2n}
        sin2_terms = (sin2_theta.unsqueeze(1)  # sin2^{n}
                      ** self.sin2_powers.unsqueeze(0))

        cos_m_theta = (self.signs.unsqueeze(0) *  # -1^{n} * C_m{2n} * cos^{m - 2n} * sin2^{n}
                       self.C_m_2n.unsqueeze(0) *
                       cos_terms *
                       sin2_terms).sum(1)  # summation of all terms

        return cos_m_theta

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight.data.t())

    def find_k(self, cos):
        # to account for acos numerical errors
        eps = 1e-7
        cos = torch.clamp(cos, -1 + eps, 1 - eps)
        acos = cos.acos()
        k = (acos / self.divisor).floor().detach()
        return k

    def forward(self, input, target=None):
        if self.training:
            assert target is not None
            x, w = input, self.weight
            beta = max(self.beta, self.beta_min)
            logit = x.mm(w)
            indexes = range(logit.size(0))
            # logit_target = logit[indexes, target.type(torch.long)]
            logit_target = torch.mean(logit[indexes] * target, dim=1)

            # cos(theta) = w * x / ||w||*||x||
            w_target_norm = w[:, target].norm(p=2, dim=0)
            x_norm = x.norm(p=2, dim=1)
            cos_theta_target = logit_target / (w_target_norm * x_norm + 1e-10)

            # equation 7
            cos_m_theta_target = self.calculate_cos_m_theta(cos_theta_target)

            # find k in equation 6
            k = self.find_k(cos_theta_target)

            # f_y_i
            logit_target_updated = (w_target_norm *
                                    x_norm *
                                    (((-1) ** k * cos_m_theta_target) - 2 * k))
            logit_target_updated_beta = (logit_target_updated + beta * logit[indexes, target]) / (1 + beta)

            logit[indexes, target] = logit_target_updated_beta
            self.beta *= self.scale
            return logit
        else:
            assert target is None
            return input.mm(self.weight)


class BCEFocalLoss(torch.nn.Module):
    """
    二分类的Focalloss alpha 固定
    """

    def __init__(self, gamma=2, alpha=0.5, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        pt = torch.sigmoid(input)
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, input, target):
        N = input.size(0)
        C = input.size(1)
        P = F.softmax(input, dim=1)
        class_mask = input.data.new(N, C).fill_(0)
        ids = target.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)
        if input.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation=None):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.activate = nn.ReLU(inplace=True)
        # self.activate = nn.CELU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activate(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activate(out)

        return out


class ResNet18(models.ResNet):
    Block = BasicBlock

    def __init__(self, in_channels, num_classes=10, **kwargs):
        Block = BasicBlock
        self.in_channels = in_channels
        super(ResNet18, self).__init__(Block, [2, 2, 2, 2], num_classes=num_classes, **kwargs)  # resnet18

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
        self.sa = Self_Attn(128)
        self.last_channels = 512 * Block.expansion
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.bnneck = nn.BatchNorm1d(self.last_channels)
        self.bnneck.bias.requires_grad_(False)  # no shift
        self.fc = torch.nn.Linear(self.last_channels, num_classes, bias=False)
        self._half = False
        self._class_normalize = True

    def init(self, model_dir=None, gain=1.):
        self.model_dir = model_dir if model_dir is not None else self.model_dir
        sd = model_zoo.load_url(model_urls['resnet18'], model_dir=self.model_dir)
        # sd = model_zoo.load_url(model_urls['resnet34'], model_dir='./models/')
        del sd['fc.weight']
        del sd['fc.bias']
        self.load_state_dict(sd, strict=False)
        torch.nn.init.xavier_uniform_(self.fc.weight, gain=gain)
        # self.lsoftmax_linear.reset_parameters()
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, inputs, targets=None, tau=8.0, reduction='avg'):  # pylint: disable=arguments-differ
        # dims = len(inputs.shape)
        # bs = inputs.shape[0]
        inputs = self.stem(inputs)
        logits = self.forward_origin(inputs, targets)
        logits /= tau

        if targets is None:
            return logits
        if targets.device != logits.device:
            targets = targets.to(device=logits.device)

        loss = self.loss_fn(input=logits, target=targets)

        if self._class_normalize and isinstance(self.loss_fn, (
                torch.nn.BCEWithLogitsLoss, skeleton.nn.BinaryCrossEntropyLabelSmooth, BCEFocalLoss)):
            # print('==========================================')
            # print(loss)
            pos = (targets == 1).to(logits.dtype)
            # print(pos)
            neg = (targets < 1).to(logits.dtype)
            # print(neg)
            npos = pos.sum(dim=0)
            # print(npos)
            nneg = neg.sum(dim=0)
            # print(nneg)

            # positive_ratio = max(0.1, min(0.9, (npos) / (npos + nneg)))
            positive_ratio = torch.clamp((npos) / (npos + nneg), min=0.03, max=0.97).view(1, loss.shape[1])
            # print(positive_ratio)
            # negative_ratio = max(0.1, min(0.9, (nneg) / (npos + nneg)))
            negative_ratio = torch.clamp((nneg) / (npos + nneg), min=0.03, max=0.97).view(1, loss.shape[1])
            # print(negative_ratio)
            LOGGER.debug('[BCEWithLogitsLoss] positive_ratio:%f, negative_ratio:%f',
                         positive_ratio, negative_ratio)

            normalized_loss = (loss * pos) / positive_ratio
            normalized_loss += (loss * neg) / negative_ratio

            loss = normalized_loss
        # print(loss)
        # loss, _ = loss.topk(k=(bs // 2))
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


class new_BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(new_BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        conv1 = conv3x3(inplanes, planes, stride)
        bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        conv2 = conv3x3(planes, planes)
        bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.res_block_pre = nn.Sequential(OrderedDict([('conv', conv1), ('bn', bn1), ('relu', self.relu)]))
        self.res_block_after = nn.Sequential(OrderedDict([('conv', conv2), ('bn', bn2)]))

    def forward(self, x):
        identity = x

        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)
        out = self.res_block_pre(x)

        # out = self.conv2(out)
        # out = self.bn2(out)
        out = self.res_block_after(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class new_ResNet18(models.ResNet):
    Block = new_BasicBlock

    def __init__(self, in_channels, num_classes=10, **kwargs):
        Block = new_BasicBlock
        super(new_ResNet18, self).__init__(Block, [2, 2, 2, 2], num_classes=num_classes, **kwargs)  # resnet18

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
        block = new_BasicBlock
        layers = [2, 2, 2, 2]
        groups = 1
        width_per_group = 64
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.last_channels = 512 * Block.expansion
        self.base_width = width_per_group
        conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                          bias=False)
        bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.conv2 = nn.Conv2d(self.inplanes, self.inplanes, kernel_size = 3, stride = 2, padding = 1)
        self.pre = nn.Sequential(
            OrderedDict([('conv1', conv1), ('bn', bn1), ('relu', self.relu), ('pool', self.maxpool)]))
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])
        self.fc = torch.nn.Linear(self.last_channels, num_classes, bias=False)
        self._half = False
        self._class_normalize = True

    def init(self, model_dir=None, gain=1.):
        # self.model_dir = model_dir if model_dir is not None else self.model_dir
        # sd = model_zoo.load_url(model_urls['resnet18'], model_dir=self.model_dir)
        # sd = model_zoo.load_url(model_urls['resnet34'], model_dir='./models/')
        # self.model_dir = model_dir if model_dir is not None else self.model_dir
        # sd = torch.load(self.model_dir + '/checkpoint.pth.tar')
        # print(sd)
        # print(sd['state_dict'])
        # new_sd = copy.deepcopy(sd['state_dict'])
        # for key,value in sd['state_dict'].items():
        #     new_sd[key[7:]] = sd['state_dict'][key]
        # del new_sd['fc.weight']
        # del new_sd['fc.bias']
        # self.load_state_dict(new_sd, strict=False)
        # torch.nn.init.xavier_uniform_(self.fc.weight, gain=gain)
        # LOGGER.debug('initialize classifier weight')

        # self.load_state_dict(sd, strict=False)
        # torch.nn.init.xavier_uniform_(self.fc.weight, gain=gain)
        # self.lsoftmax_linear.reset_parameters()
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
            self.stem = torch.nn.Sequential(
                # skeleton.nn.Permute(0, 3, 1, 2),
                skeleton.nn.Normalize(0.5 * 255, 0.25 * 255, inplace=False),
                torch.nn.Conv2d(self.in_channels, 3, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(3),
            ).cuda()

    def forward_origin(self, x, targets):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, inputs, targets=None, tau=8.0, reduction='avg'):  # pylint: disable=arguments-differ
        # dims = len(inputs.shape)
        # bs = inputs.shape[0]
        inputs = self.stem(inputs)
        logits = self.forward_origin(inputs, targets)
        logits /= tau

        if targets is None:
            return logits
        if targets.device != logits.device:
            targets = targets.to(device=logits.device)

        loss = self.loss_fn(input=logits, target=targets)

        if self._class_normalize and isinstance(self.loss_fn, (
                torch.nn.BCEWithLogitsLoss, skeleton.nn.BinaryCrossEntropyLabelSmooth, BCEFocalLoss)):
            # print('==========================================')
            # print(loss)
            pos = (targets == 1).to(logits.dtype)
            # print(pos)
            neg = (targets < 1).to(logits.dtype)
            # print(neg)
            npos = pos.sum(dim=0)
            # print(npos)
            nneg = neg.sum(dim=0)
            # print(nneg)

            # positive_ratio = max(0.1, min(0.9, (npos) / (npos + nneg)))
            positive_ratio = torch.clamp((npos) / (npos + nneg), min=0.03, max=0.97).view(1, loss.shape[1])
            # print(positive_ratio)
            # negative_ratio = max(0.1, min(0.9, (nneg) / (npos + nneg)))
            negative_ratio = torch.clamp((nneg) / (npos + nneg), min=0.03, max=0.97).view(1, loss.shape[1])
            # print(negative_ratio)
            LOGGER.debug('[BCEWithLogitsLoss] positive_ratio:%f, negative_ratio:%f',
                         positive_ratio, negative_ratio)

            normalized_loss = (loss * pos) / positive_ratio
            normalized_loss += (loss * neg) / negative_ratio

            loss = normalized_loss
        # print(loss)
        # loss, _ = loss.topk(k=(bs // 2))
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


class ResLayer(nn.Module):
    def __init__(self, in_c, out_c, groups=1):
        super(ResLayer, self).__init__()
        self.act = nn.CELU(0.075, inplace=False)
        conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                         bias=False, groups=groups)
        norm = nn.BatchNorm2d(num_features=out_c)
        pool = nn.MaxPool2d(2)
        self.pre_conv = nn.Sequential(
            OrderedDict([('conv', conv), ('pool', pool), ('norm', norm), ('act', nn.CELU(0.075, inplace=False))]))
        self.res1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               bias=False, groups=groups)), ('bn', nn.BatchNorm2d(out_c)),
            ('act', nn.CELU(0.075, inplace=False))]))
        self.res2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               bias=False, groups=groups)), ('bn', nn.BatchNorm2d(out_c)),
            ('act', nn.CELU(0.075, inplace=False))]))

    def forward(self, x):
        # x = self.conv(x)
        # x = self.norm(x)
        # x = self.act(x)
        # x = self.pool(x)
        x = self.pre_conv(x)
        out = self.res1(x)
        out = self.res2(out)
        out = out + x
        return out


class ResNet9(nn.Module):
    # Block = BasicBlock

    def __init__(self, in_channels, num_classes=10, **kwargs):
        # Block = BasicBlock
        super(ResNet9, self).__init__()  # resnet18
        channels = [64, 128, 256, 512]
        group = 1
        self.in_channels = in_channels
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
        conv1 = nn.Conv2d(in_channels=3, out_channels=channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                          bias=False)
        norm1 = nn.BatchNorm2d(num_features=channels[0])
        act = nn.CELU(0.075, inplace=False)
        pool = nn.MaxPool2d(2)
        self.prep = nn.Sequential(OrderedDict([('conv', conv1), ('bn', norm1), ('act', act)]))
        self.layer1 = ResLayer(channels[0], channels[1], groups=group)
        conv2 = nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1),
                          bias=False, groups=group)
        norm2 = nn.BatchNorm2d(num_features=channels[2])
        self.layer2 = nn.Sequential(OrderedDict([('conv', conv2), ('pool', pool), ('bn', norm2), ('act', act)]))
        self.layer3 = ResLayer(channels[2], channels[3], groups=group)
        # self.pool4 = nn.MaxPool2d(4)
        self.pool4 = nn.AdaptiveMaxPool2d(1)
        # self.conv3 = nn.Linear(512, num_classes, bias=False)
        self.fc = torch.nn.Linear(channels[3], num_classes, bias=False)
        self._half = False
        self._class_normalize = True

    def init(self, model_dir=None, gain=1.):
        self.model_dir = model_dir if model_dir is not None else self.model_dir
        # sd = model_zoo.load_url(model_urls['resnet18'], model_dir=self.model_dir)
        sd = model_zoo.load_url(
            'https://github.com/paperscodes/test/releases/download/test/r9.pth.tar',
            model_dir=self.model_dir)
        # self.model_dir = model_dir if model_dir is not None else self.model_dir
        # sd = torch.load(self.model_dir + '/resnet9.pth.tar')
        # for m in self.prep:
        #     m.requires_grad_(False)  # no shift
        # print(sd)
        # print(sd['state_dict'])
        new_sd = copy.deepcopy(sd['state_dict'])
        for key, value in sd['state_dict'].items():
            new_sd[key[7:]] = sd['state_dict'][key]
        # del new_sd['fc.weight']
        # del new_sd['fc.bias']
        self.load_state_dict(new_sd, strict=False)
        # for m in self.modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # torch.nn.init.xavier_uniform_(self.fc.weight, gain=gain)
        # LOGGER.debug('initialize classifier weight')

        # self.load_state_dict(sd, strict=False)
        # torch.nn.init.xavier_uniform_(self.fc.weight, gain=gain)
        # self.lsoftmax_linear.reset_parameters()
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
            self.stem = torch.nn.Sequential(
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
        # dims = len(inputs.shape)
        # bs = inputs.shape[0]
        inputs = self.stem(inputs)
        logits = self.forward_origin(inputs, targets)
        logits /= tau

        if targets is None:
            return logits
        if targets.device != logits.device:
            targets = targets.to(device=logits.device)

        loss = self.loss_fn(input=logits, target=targets)

        if self._class_normalize and isinstance(self.loss_fn, (
                torch.nn.BCEWithLogitsLoss, skeleton.nn.BinaryCrossEntropyLabelSmooth, BCEFocalLoss)):
            # print('==========================================')
            # print(loss)
            pos = (targets == 1).to(logits.dtype)
            # print(pos)
            neg = (targets < 1).to(logits.dtype)
            # print(neg)
            npos = pos.sum(dim=0)
            # print(npos)
            nneg = neg.sum(dim=0)
            # print(nneg)

            # positive_ratio = max(0.1, min(0.9, (npos) / (npos + nneg)))
            positive_ratio = torch.clamp((npos) / (npos + nneg), min=0.03, max=0.97).view(1, loss.shape[1])
            # print(positive_ratio)
            # negative_ratio = max(0.1, min(0.9, (nneg) / (npos + nneg)))
            negative_ratio = torch.clamp((nneg) / (npos + nneg), min=0.03, max=0.97).view(1, loss.shape[1])
            # print(negative_ratio)
            LOGGER.debug('[BCEWithLogitsLoss] positive_ratio:%f, negative_ratio:%f',
                         positive_ratio, negative_ratio)

            normalized_loss = (loss * pos) / positive_ratio
            normalized_loss += (loss * neg) / negative_ratio

            loss = normalized_loss
        # print(loss)
        # loss, _ = loss.topk(k=(bs // 2))
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
