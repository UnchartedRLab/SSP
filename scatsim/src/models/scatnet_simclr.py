from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from kymatio.torch import Scattering2D


def conv3x3(in_planes: int, out_planes: int, stride: int = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, scatchannels, inchannels, in_planes):
        super(ResNet, self).__init__()
        self.in_planes = in_planes

        self.conv1 = nn.Conv2d(scatchannels, inchannels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inchannels)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        #self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = out.view(out.size(0), -1)
        #out = F.avg_pool1d(out, 2)
        
        #out = self.linear(out)
        return out


def ResNet50(scatchannels, inchannels, in_planes):
    return ResNet(Bottleneck, [3, 4, 6, 3], scatchannels, inchannels, in_planes)



class ScatSimCLR(nn.Module):

    def __init__(self, J: int, L: int, input_size: Tuple[int, int, int], res_blocks: int, out_dim: int):
        super(ScatSimCLR, self).__init__()

        if J < 1:
            raise ValueError('Incorrect `J` parameter')

        if L < 1:
            raise ValueError('Incorrect `L` parameter')

        if len(input_size) != 3:
            raise ValueError('`input_size` parameter should be (H, W, C)')

        self._J = J
        self._L = L

        self._res_blocks = res_blocks
        self._out_dim = out_dim

        # get image height, width and channels
        h, w, c = input_size
        # ScatNet is applied for each image channel separately
        self._num_scatnet_channels = c * ((L * L * J * (J - 1)) // 2 + L * J + 1)

        # max order is always 2 - maximum possible
        self._scatnet = Scattering2D(J=J, shape=(h, w), L=L, max_order=2)

        # batch size, which is applied to ScatNet features
        self._scatnet_bn = nn.BatchNorm2d(self._num_scatnet_channels)

        # pool size in adapter network
        self._pool_size = 4

        self.inplanes = 256 #self.INPLANES[self._res_blocks]
        # adapter network
        self._adapter_network = ResNet50(self._num_scatnet_channels, 256, self.inplanes)

        # linear layers
        num_ftrs = 2048
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

  














    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scatnet = self._scatnet(x).squeeze(1)

        B, C, FN, H, W = scatnet.size()
        scatnet = scatnet.view(B, C * FN, H, W)

        h = self._adapter_network(scatnet)
        h = h.view(h.size(0), -1)

        z = self.l2(F.relu(self.l1(h)))
        return h, z
