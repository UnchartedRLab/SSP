import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


import torch.nn as nn

model = torch.hub.load('yukimasano/PASS:main', 'moco_resnet50')

transform = transforms.Compose(
    [transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()]
)
dataset_train = datasets.ImageFolder("train", transform=transform)
dataset_test = datasets.ImageFolder("validation", transform=transform)


batch_size = 100
trainloader = torch.utils.data.DataLoader(
    dataset_train, batch_size=batch_size
)
testloader = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size
)


import numpy as np
with torch.no_grad():
    features_train = []
    train_labels = []
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, label= data
        # forward + backward + optimize
        outputs = np.array(model(inputs))
        features_train.append((outputs))
        train_labels.append(label)
        print(i)
        if i == 0:
            print(outputs.shape)
        if (i % 1000 == 0) and i > 0:
            train_labels = np.concatenate(train_labels, axis=0)
            features_train = np.concatenate(features_train, axis=0)
            np.savez_compressed('./new_imagenet_resnet50_800_pass_train_{}'.format(i), train_data=features_train, train_labels=train_labels)
            features_train = []
            train_labels = []
            print("saved to imagenet_resnet50_800_pass_train_{}".format(i))


    if len(features_train)  != 0:
        train_labels = np.concatenate(train_labels, axis=0)
        features_train = np.concatenate(features_train, axis=0)
        np.savez_compressed('./new_imagenet_resnet50_800_pass_train_last', train_data=features_train, train_labels=train_labels)
        features_train = []
        train_labels = []
    print('Finished Training')


    features_test = []
    test_labels = []
    for i, data in enumerate(testloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, label = data
        # forward + backward + optimize
        outputs = np.array(model(inputs))
    
        features_test.append((outputs))
        test_labels.append(label)
        if (i % 1000 == 0) and i > 0:
            test_labels = np.concatenate(test_labels, axis=0)
            features_test = np.concatenate(features_test, axis=0)
            np.savez_compressed('./new_imagenet_resnet50_800_pass_test_{}'.format(i), test_data=features_test, test_labels=test_labels)
            features_test = []
            test_labels = []
    if len(features_test)  != 0:
        test_labels = np.concatenate(test_labels, axis=0)
        features_test = np.concatenate(features_test, axis=0)
        np.savez_compressed('./new_imagenet_resnet50_800_pass_test_last', test_data=features_test, test_labels=test_labels)
        features_test= []
        test_labels = []
    print('Finished Testing')
