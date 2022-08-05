import os

import bz2
import pickle
import _pickle as cPickle
from PIL import Image

from torchvision import datasets
from torch.utils.data import Dataset
from .cifar_20 import CIFAR20

from sklearn.model_selection import train_test_split



SUPPORTED_DATASETS = ['stl10', 'cifar10', 'cifar20', 'cifar100', 'single', 'mini', 'plant', 'eurosat', 'isic2018']

IMAGENET_STATS = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

CIFAR_STATS = {
    'mean': [0.491, 0.482, 0.447],
    'std': [0.247, 0.243, 0.261]
}

DATASET_STATS = {
    'cifar10': CIFAR_STATS,
    'cifar20': CIFAR_STATS,
    'cifar100': CIFAR_STATS,
    'single': CIFAR_STATS,
    'stl10': IMAGENET_STATS,
    'mini': IMAGENET_STATS,
    'plant': IMAGENET_STATS,
    'eurosat': IMAGENET_STATS,
    'isic2018': IMAGENET_STATS,
}

NUM_CLASSES = {
    'cifar10': 10,
    'cifar20': 20,
    'cifar100': 100,
    'single' : 10,
    'stl10': 10,
    'mini': 64,
    'plant': 38,
    'eurosat': 10,
    'isic2018': 7,
}


def load_pickle_data(file):
    with open(file, "rb") as fo:
        data = pickle.load(fo, encoding="bytes")
        if isinstance(data, dict):
            data = {k.decode("ascii"): v for k, v in data.items()}

    return data


def build_label_index(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)
        
    return label2inds




class MiniImageNetBase(Dataset):
    def __init__(self, train=True, transform=None):
        print("mini dataset starts")
        self.transform = transform
        train_file = './compresseddatasets/mini/miniImageNet_category_split_train_phase_train.pickle' 
        test_file = './compresseddatasets/mini/miniImageNet_category_split_train_phase_test.pickle' 

        #val_file = './compresseddatasets/mini/miniImageNet_category_split_train_phase_val.pickle' 
        #val2_file = './compresseddatasets/mini/miniImageNet_category_split_val.pickle' 
        #test2_file = './compresseddatasets/mini/miniImageNet_category_split_test.pickle' 
        if train:
            path_me = train_file
        else:
            path_me = test_file

        loaded = load_pickle_data(path_me)
        self.data = loaded["data"]
        self.labels = loaded["labels"]
        print("mini dataset done!")  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)

        return image, label





class PlantData(Dataset):
    def __init__(self, train=True, transform=None):
        print("plant dataset starts")
        self.transform = transform
        train_label_file = './compresseddatasets/plant2/plant2_train_label.pbz2' 
        train_data_file =  './compresseddatasets/plant2/plant2_train_data.pbz2' 
        test_label_file =  './compresseddatasets/plant2/plant2_test_label.pbz2' 
        test_data_file =   './compresseddatasets/plant2/plant2_test_data.pbz2' 

        if train:
            label_file = train_label_file
            data_file = train_data_file
        else:
            label_file = test_label_file
            data_file = test_data_file

        loaded_labels = bz2.BZ2File(label_file, 'rb')
        self.labels = cPickle.load(loaded_labels)

        loaded_images = bz2.BZ2File(data_file, 'rb')
        self.data = cPickle.load(loaded_images)

        print("plant dataset done!")  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx] - 1
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)

        return image, label




class ISICData(Dataset):
    def __init__(self, train=True, transform=None):
        print("isic2018 dataset starts")
        self.transform = transform
        train_label_file = './compresseddatasets/ISIC2018/ISIC2018_train_label.pbz2' 
        train_data_file =  './compresseddatasets/ISIC2018/ISIC2018_train_data.pbz2' 
        test_label_file =  './compresseddatasets/ISIC2018/ISIC2018_test_label.pbz2' 
        test_data_file =   './compresseddatasets/ISIC2018/ISIC2018_test_data.pbz2' 

        if train:
            label_file = train_label_file
            data_file = train_data_file
        else:
            label_file = test_label_file
            data_file = test_data_file

        loaded_labels = bz2.BZ2File(label_file, 'rb')
        self.labels = cPickle.load(loaded_labels)

        loaded_images = bz2.BZ2File(data_file, 'rb')
        self.data = cPickle.load(loaded_images)

        print("isic2018 dataset done!")  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx] - 1
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)

        return image, label





class EuroSATData(Dataset):
    def __init__(self, train=True, transform=None):
        print("eurosat dataset starts")
        self.transform = transform
        train_label_file = './compresseddatasets/eurosat_train_label.pbz2' 
        train_data_file =  './compresseddatasets/eurosat_train_data.pbz2' 
        test_label_file =  './compresseddatasets/eurosat_test_label.pbz2' 
        test_data_file =   './compresseddatasets/eurosat_test_data.pbz2' 

        if train:
            label_file = train_label_file
            data_file = train_data_file
        else:
            label_file = test_label_file
            data_file = test_data_file

        loaded_labels = bz2.BZ2File(label_file, 'rb')
        self.labels = cPickle.load(loaded_labels)

        loaded_images = bz2.BZ2File(data_file, 'rb')
        self.data = cPickle.load(loaded_images)

        print("eurosat dataset done!")  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx] - 1
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)

        return image, label




class ADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        print("dataset starts")
        self.root_dir = root_dir
        self.transform = transform
        img_data = bz2.BZ2File(self.root_dir, 'rb')
        self.img_data = cPickle.load(img_data)
        print("dataset done!")

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        image = self.img_data[idx]
        if self.transform is not None:
            image = self.transform(self.img_data[idx])

        return image, 0



def get_dataset(dataset: str, train: bool,
                transform=None,
                download: bool = False,
                unlabeled: bool = False) -> Dataset:

    if dataset not in SUPPORTED_DATASETS:
        raise ValueError('Unsupported dataset')

    if dataset == 'stl10':

        if train and unlabeled:
            split = 'train+unlabeled'
        elif train:
            split = 'train'
        elif unlabeled:
            split = 'unlabeled'
        else:
            split = 'test'

        return datasets.STL10('/content/gdrive/MyDrive/effslm/data', split=split, download=download, transform=transform)
    elif dataset == 'cifar10':
        return datasets.CIFAR10('/content/gdrive/MyDrive/effslm/data', train=train, download=download, transform=transform)
    elif dataset == 'cifar100':
        return datasets.CIFAR100('/content/gdrive/MyDrive/effslm/data', train=train, download=download, transform=transform)
    elif dataset == 'cifar20':
        return CIFAR20('./compresseddatasets/cifar-20', train=train, download=download, transform=transform)
    elif dataset == 'single':
        return ADataset('./compresseddatasets/single/single84.pbz2', transform=transform)
    elif dataset == 'mini':
        return MiniImageNetBase(train = train, transform=transform)
    elif dataset == 'plant':
        return PlantData(train = train, transform=transform)
    elif dataset == 'eurosat':
        return EuroSATData(train = train, transform=transform)
    elif dataset == 'isic2018':
        return ISICData(train = train, transform=transform)

