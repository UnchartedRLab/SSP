from numpy import extract
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from src.data import datasets
from src.models import ResNetSimCLR, ScatSimCLR, HarmSimCLR
from src.data.datasets import ISICData, PlantData, EuroSATData
import numpy as np
import argparse


def extract(checkpoint, trainset, testset, dataset_name):
    model = HarmSimCLR(128)
    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict, strict=False)
    print(f'Loaded: {checkpoint}')
    model.eval()

    trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=100,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, num_workers=2, pin_memory=True
    )

    device = "cuda"
    model.to(device)

    with torch.no_grad():
        features_train = []
        train_labels = []
        for i, data in enumerate(trainloader, 0):
            inputs, label= data
            inputs = inputs.to(device)
            outputs = model(inputs)[0]

            outputs = np.array(outputs.to('cpu'))
            features_train.append((outputs))
            train_labels.append(label)
        print('Finished Training Set')

        features_train = np.concatenate(features_train, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        print(features_train.shape)

        features_test = []
        test_labels = []

        for i, data in enumerate(testloader, 0):
            inputs, label= data
            inputs = inputs.to(device)
            outputs = model(inputs)[0]
            outputs = np.array(outputs.to('cpu'))
            features_test.append((outputs))
            test_labels.append(label)
        print('Finished Testing Set')
        features_test = np.concatenate(features_test, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)
        print(features_test.shape)

        np.savez_compressed('./features_{}_{}'.format(checkpoint, dataset_name), train_data=features_train, test_data=features_test, train_labels=train_labels, test_labels=test_labels)








def main(args):

    dataset = args.dataset
    checkpoint_file = args.checkpoint_path
    transform = transforms.Compose([
                transforms.Resize(84),
                transforms.CenterCrop(84),
                transforms.ToTensor()
    ])

    if dataset == "isic2018":
        train = ISICData(True, transform=transform)
        test = ISICData(False, transform=transform)

    elif dataset == 'plants':
        train = PlantData(True, transform=transform)
        test = PlantData(False, transform=transform)
    
    elif dataset == 'eurosat':
        train = EuroSATData(True, transform=transform)
        test = EuroSATData(False, transform=transform)
    else:
        print("Unsupported Dataset")
        
    extract(checkpoint_file, train, test, dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-m',
                        help='datast to extract features from',
                        choices=['isic2018', 'plants', 'eurosat', 'cifar10'])
    parser.add_argument('--checkpoint_path', '-c',
                        help='Path to checkpoint file')
    args = parser.parse_args()
    main(args)




