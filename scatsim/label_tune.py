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


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return Harm2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation, use_bn=False, level=None, DC=True, groups=groups)

    #return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                 padding=dilation, groups=groups, bias=False, dilation=dilation)


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
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
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
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
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

    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = Harm2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False, use_bn=True)
        #nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 512 * block.expansion

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #x = torch.flatten(x, 1)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


class HarmSimCLR(nn.Module):

    def __init__(self, out_dim: int):
        super(HarmSimCLR, self).__init__()

        self._out_dim = out_dim

        # pool size in adapter network
        self._pool_size = 4

        # adapter network
        self._adapter_network = resnet50()

        # linear layers
        num_ftrs = 512 * 4
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self._adapter_network(x)
        h = h.squeeze()

        z = self.l1(h)
        z = F.relu(z)
        z = self.l2(z)

        return h, z


class FineTuneMe(nn.Module):
    def __init__(self, class_num: int):
        super(FineTuneMe, self).__init__()

        self.head = HarmSimCLR(128)
        self.fc = nn.Linear(2048, class_num)

    def load_weight(self, checkpoints_file):
        state_dict = torch.load(checkpoints_file)
        self.head.load_state_dict(state_dict)

    def get_head(self):
        return self.head

    def save_head(self, model_path):
        torch.save(self.head.state_dict(), model_path)

    def forward(self, x: torch.Tensor):
        x, _ = self.head(x)
        x = F.relu(x)
        x = self.fc(x)

        return x




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
            h, _ = self._model(batch_x)
            features.extend(h.cpu().detach().numpy())

        features = np.array(features)
        labels = np.array(labels)
        return features, labels











def cross_entropy(logits, one_hot_targets):
    logsoftmax_fn = nn.LogSoftmax(dim=1)
    logsoftmax = logsoftmax_fn(logits)
    return - (one_hot_targets * logsoftmax).sum(1).mean()


def smooth_one_hot(targets, num_classes):
    device = torch.device("cuda:0")
    label_smoothing = 0.1
    with torch.no_grad():
        new_targets = torch.empty(size=(targets.size(0), num_classes), device=device)
        new_targets.fill_(label_smoothing / (num_classes-1))
        new_targets.scatter_(1, targets.unsqueeze(1), 1. - label_smoothing)
    return new_targets





def prepare_dataset(dataset):
    transform = ValidAugmentor(dataset, [84, 84, 3])
    train_dataset = get_dataset(dataset=dataset, train=True, transform=transform)
    test_dataset = get_dataset(dataset=dataset, train=False, transform=transform)
    return train_dataset, test_dataset


def make_model(dataset, checkpoint):
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

    model = FineTuneMe(n_outputs)
    model.load_weight(checkpoint)

    return model



def evaluation(dataset, checkpoint, batch_size):
    device = torch.device("cuda:0")
    model = HarmSimCLR(128)
    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict)
    
    
    
    model=make_model(dataset, checkpoint)
    model=model.to(device)
    testEmbedding = EmbeddingExtractor(model.get_head(), dataset=dataset, input_size=[84, 84, 3], batch_size=batch_size)
    train_data, train_labels, test_data, test_labels = testEmbedding.get_features()
    evaluator = LogisticRegressionEvaluator(n_features=train_data.shape[1], n_classes=NUM_CLASSES[dataset], device=device, batch_size=batch_size)
    accuracy = evaluator.run_evaluation(train_data, train_labels, test_data, test_labels, 30)
    
    print(accuracy)

    return accuracy




def map_fn(index, flags):
    device = torch.device("cuda:0")
    train_datasetn = flags['train_dataset']

    train_dataset, test_dataset = prepare_dataset(train_datasetn) 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=flags['batch_size'], shuffle=True, num_workers=4, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=flags['batch_size'], shuffle=False, num_workers=4, drop_last=False)

    model = make_model(train_datasetn, flags['checkpoint'])
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9, weight_decay = 5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

    def train_loop_fn(model, loader):
        model = model.to(device)
        model.train()
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            smoothed_targets = smooth_one_hot(target, 64)
            output = model(data)
            loss = cross_entropy(output, smoothed_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
          

    def test_loop_fn(model, loader):
        model = model.to(device)
        total_samples = 0
        correct = 0
        model.eval()
        data, pred, target = None, None, None
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size()[0]
        
        accuracy = 100.0 * correct / total_samples
        return accuracy

    best_acc = 0.0
    accuracy = 0.0
    for epoch in range(0, flags['total_epoch']):
        print("start training epoch {}".format(epoch))
        train_loop_fn(model, train_loader)
        
        if epoch % flags['test_steps'] == 0:
            accuracy = test_loop_fn(model, test_loader)
            print(accuracy)
            if accuracy > best_acc:
                best_acc = accuracy
                model_path = f'super_{train_datasetn}_{epoch}.pth'
                model.save_head(model_path)
        if epoch > 9:
            scheduler.step()




                

   
    


if __name__ == '__main__':
    flags = {}
    flags['train_dataset'] = 'mini'
    flags['test_dataset'] = 'plant'
    flags['checkpoint'] = '/content/gdrive/MyDrive/scatsim/mini_200.pth'
    flags['batch_size'] = 128
    flags['log_steps'] = 1024
    flags['test_steps'] = 5
    flags['total_epoch'] = 300
    map_fn(1, flags)

    #acc = evaluation(flags['test_dataset'], flags['checkpoint'], flags['batch_size'])
    #print(acc)

    