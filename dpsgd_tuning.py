import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from opacus import PrivacyEngine

from lib.train_tools import get_device, train_dpsgd, test
from lib.data import get_feature_data, get_feature_data_imagenet
from lib.log import Logger
import pandas as pd


class Net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.input_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels//16, 256, 4)
        self.fc1 = nn.Linear(256, out_channels)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1]//16, 4, 4)
        x = F.relu(self.conv1(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        return x


class FullyConnected(nn.Module):
    def __init__(
        self,
        hidden_size,
        input_num,
        output_num,
    ):
        super(FullyConnected, self).__init__()
        self.fc1 = nn.Linear(input_num, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_num)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=0)
        return x


def main(dataset='cifar10', batch_size=2048, physical_batch_size=256, feature_path=None,
         lr=1, delta=1e-5, optim="SGD", momentum=0.9, nesterov=False, noise_multiplier=1,
         max_grad_norm=0.1, epochs=100, gamma=0.9, hidden_size=512, logdir=None, extractor='vit'):

    logger = Logger(logdir)
    device = get_device()

    if dataset == 'cifar10':
        n_outputs=10
    elif dataset == 'plant':
        n_outputs=39
    elif dataset == 'eurosat':
        n_outputs=10
    elif dataset == 'isic2018':
        n_outputs=7
    elif dataset == 'cifar100':
        n_outputs=100
    elif dataset == 'mnist':
        n_outputs=10
    elif dataset == 'fmnist':
        n_outputs=10
    elif dataset == 'imagenet':
        n_outputs=1000
    else:
        n_outputs=100

    assert batch_size % physical_batch_size == 0
    if dataset == 'imagenet':
        test_loader, train_loader, n_features, n_train = get_feature_data_imagenet(feature_path, batch_size, False, 1, False)
    else:
        test_loader, train_loader, n_features, n_train = get_feature_data(feature_path, batch_size, False, 1, False)
    
    model = nn.Sequential(nn.Linear(n_features, n_outputs)).to(device)
    # model = FullyConnected(hidden_size, n_features, n_outputs).to(device)


    privacy_engine = PrivacyEngine()

    if optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=momentum,
                                    nesterov=nesterov)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # scheduler = StepLR(optimizer, step_size=1, gamma=0.4, verbose=True)

    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        max_grad_norm=max_grad_norm,
        noise_multiplier=noise_multiplier,
    )

    epsilons = []
    test_accs = []
    train_accs = []
    test_losss = []
    train_losss = []

    #"criterion" or "no"
    for epoch in range(0, epochs):
        #print('Epoch:', epoch,', LR:', scheduler.get_last_lr())
        train_loss, train_acc, epsilon = train_dpsgd(model, train_loader, physical_batch_size, 
                                               optimizer, privacy_engine, True, delta, epoch, 
                                               device)
        test_loss, test_acc = test(model, test_loader, True, epoch, device)
        print(f"{epsilon},{test_acc}")
        #logger.log_epoch(epoch, train_loss, train_acc, test_loss, test_acc, epsilon)
        # scheduler.step()
        epsilons.append(epsilon)
        test_accs.append(test_acc)
        train_losss.append(train_loss)
        train_accs.append(train_acc)
        test_losss.append(test_loss)


    results_file_name = './results/{}_dpsgd_{}.csv'.format(dataset, extractor)

    a = np.array(epsilons)
    b = np.array(test_accs)
    c = np.array(train_accs)
    d = np.array(test_losss)
    e = np.array(train_losss)

    df = pd.DataFrame({"epsilon" : a, "test_acc" : b, 'train_acc' : c, 'train_loss' : e, 'test_loss' : d})
    df.to_csv(results_file_name, index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cifar100', 'cifar10', 'fmnist', 'mnist', 'plant', 'eurosat', 'isic2018'])
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--physical-batch-size', type=int, default=256)
    parser.add_argument('--hidden-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--optim', type=str, default="SGD", choices=["SGD", "Adam"])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', action="store_true")
    parser.add_argument('--noise-multiplier', type=float, default=1)
    parser.add_argument('--max-grad-norm', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--feature-path', default=None)
    parser.add_argument('--logdir', default=None)
    parser.add_argument('--extractor', default='vit')
    args = parser.parse_args()
    main(**vars(args))
