import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
import math
import numpy as np

import torch.optim as optim


from src.data.datasets import get_dataset
from src.data.augmentor import ValidAugmentor
from src.evaluation import LogisticRegressionEvaluator
from src.data import EmbeddingExtractor
from src.data.datasets import NUM_CLASSES


__all__ = ['resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, feature=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.fc is None:
            if feature:
                return x, None
            else:
                return x
        if feature:
            x1 = self.fc(x)
            return x, x1
        x = self.fc(x)
        return x



def resnet18(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


class EmbeddingExtractor:
    def __init__(self, model: nn.Module, dataset: str, input_size: Tuple[int, int, int], batch_size: int):
        self._model = model
        self._device = torch.device("cuda:0")
        self._dataset = dataset
        self._batch_size = batch_size
        self._transform = ValidAugmentor(self._dataset, input_size)

    def get_features(self):
        train_dataset = get_dataset(dataset=self._dataset, train=True, transform=self._transform, download=True)
        test_dataset = get_dataset(dataset=self._dataset, train=False, transform=self._transform, download=True)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self._batch_size, drop_last=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self._batch_size, drop_last=False)

        train_features, train_labels = self._compute_embeddings(train_loader)
        test_features, test_labels = self._compute_embeddings(test_loader)
        return train_features, train_labels, test_features, test_labels

    def _compute_embeddings(self, loader: torch.utils.data.DataLoader):
        features = []
        labels = []

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self._device)
            labels.extend(batch_y)
            h, _ = self._model(batch_x, True)
            features.extend(h.cpu().detach().numpy())

        features = np.array(features)
        labels = np.array(labels)
        return features, labels















def prepare_dataset(dataset):
    transform = ValidAugmentor(dataset, [84, 84, 3])
    train_dataset = get_dataset(dataset=dataset, train=True, transform=transform)
    test_dataset = get_dataset(dataset=dataset, train=False, transform=transform)
    return train_dataset, test_dataset






def evaluation(dataset, checkpoint, batch_size):
    device = torch.device("cuda:0")
    model = resnet18()
    check = torch.load(checkpoint)
    model_dict = model.state_dict()
    params = check['state_dict']
    params = {k: v for k, v in params.items() if k in model_dict}
    model_dict.update(params)
    model.load_state_dict(model_dict)

    model=model.to(device)
    testEmbedding = EmbeddingExtractor(model, dataset=dataset, input_size=[84, 84, 3], batch_size=batch_size)
    train_data, train_labels, test_data, test_labels = testEmbedding.get_features()

    if dataset == 'cifar10':
        n_outputs=10
    elif dataset == 'mini':
        n_outputs=64
    elif dataset == 'plant':
        n_outputs=38
    elif dataset == 'eurosat':
        n_outputs=10
    elif dataset == 'isic2018':
        n_outputs=7

    evaluator = LogisticRegressionEvaluator(n_features=train_data.shape[1], n_classes=n_outputs, device=device, batch_size=batch_size)
    accuracy = evaluator.run_evaluation(train_data, train_labels, test_data, test_labels, 30)

    return accuracy


if __name__ == '__main__':
    flags = {}
    flags['dataset'] = 'isic2018'
    flags['checkpoint'] = '/content/gdrive/MyDrive/scatsim/model_best.pth.tar'
    flags['batch_size'] = 128
    acc = evaluation(flags['dataset'], flags['checkpoint'], flags['batch_size'])
    print(acc)

    