from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

__all__ = ['resnet18']



def dct_filters(k=3, groups=1, expand_dim=1, level=None, DC=True, l1_norm=True):
    if level is None:
        nf = k**2 - int(not DC) 
    else:
        if level <= k:
            nf = level*(level+1)//2 - int(not DC) 
        else:
            r = 2*k-1 - level
            nf = k**2 - r*(r+1)//2 - int(not DC)
    filter_bank = np.zeros((nf, k, k), dtype=np.float32)
    m = 0
    for i in range(k):
        for j in range(k):
            if (not DC and i == 0 and j == 0) or (not level is None and i + j >= level):
                continue
            for x in range(k):
                for y in range(k):
                    filter_bank[m, x, y] = math.cos((math.pi * (x + .5) * i) / k) * math.cos((math.pi * (y + .5) * j) / k)
            if l1_norm:
                filter_bank[m, :, :] /= np.sum(np.abs(filter_bank[m, :, :]))
            else:
                ai = 1.0 if i > 0 else 1.0 / math.sqrt(2.0)
                aj = 1.0 if j > 0 else 1.0 / math.sqrt(2.0)
                filter_bank[m, :, :] *= (2.0 / k) * ai * aj
            m += 1
    filter_bank = np.tile(np.expand_dims(filter_bank, axis=expand_dim), (groups, 1, 1, 1))
    return torch.FloatTensor(filter_bank)


class Harm2d(nn.Module):

    def __init__(self, ni, no, kernel_size, stride=1, padding=0, bias=True, dilation=1, use_bn=False, level=None, DC=True, groups=1):
        super(Harm2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.dct = nn.Parameter(dct_filters(k=kernel_size, groups=ni if use_bn else 1, expand_dim=1 if use_bn else 0, level=level, DC=DC), requires_grad=False)
        
        nf = self.dct.shape[0] // ni if use_bn else self.dct.shape[1]
        if use_bn:
            self.bn = nn.BatchNorm2d(ni*nf, affine=False)
            self.weight = nn.Parameter(nn.init.kaiming_normal_(torch.Tensor(no, ni // self.groups * nf, 1, 1), mode='fan_out', nonlinearity='relu'))
        else:
            self.weight = nn.Parameter(nn.init.kaiming_normal_(torch.Tensor(no, ni // self.groups, nf, 1, 1), mode='fan_out', nonlinearity='relu'))
        self.bias = nn.Parameter(nn.init.zeros_(torch.Tensor(no))) if bias else None

    def forward(self, x):
        if not hasattr(self, 'bn'):
            filt = torch.sum(self.weight * self.dct, dim=2)
            x = F.conv2d(x, filt, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
            return x
        else:
            x = F.conv2d(x, self.dct, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=x.size(1))
            x = self.bn(x)
            x = F.conv2d(x, self.weight, bias=self.bias, padding=0, groups=self.groups)
            return x


def conv3x3(in_planes, out_planes, stride=1, level=None):
    return Harm2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, use_bn=False, level=level)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.GroupNorm(32, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.GroupNorm(32, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = Harm2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False, use_bn=True)
        #nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(32, 64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.GroupNorm(32, planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x



def resnet18():
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    return model




class HarmSimCLR(nn.Module):
    def __init__(self, out_dim: int):
        super(HarmSimCLR, self).__init__()
        self._out_dim = out_dim
        self._pool_size = 4
        self._adapter_network = resnet18()
        num_ftrs = 512
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self._adapter_network(x)
        h = h.squeeze()
        z = self.l1(h)
        z = F.relu(z)
        z = self.l2(z)

        return h, z