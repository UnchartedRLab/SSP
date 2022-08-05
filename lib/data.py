import os
import pickle
import numpy as np
import logging

import torch
from torchvision import datasets, transforms
from sklearn.utils import shuffle

SHAPES = {
    "cifar10": (32, 32, 3),
    "cifar10_500K": (32, 32, 3),
    "fmnist": (28, 28, 1),
    "mnist": (28, 28, 1)
}



def get_feature_data_imagenet(feature_path, batch_size, shu, worker, pin=True):

    train_path="./features/imagenet_train.npz"
    train_file = np.load(train_path)

    test_path="./features/imagenet_test.npz"
    test_file = np.load(test_path)
    
    names = []
    for k in train_file.files:
        names.append(k)

    if "train_label" in names:
        train_labels = 'train_label'
    else:
        train_labels = 'train_labels'

    names = []
    for k in test_file.files:
        names.append(k)

    if "test_label" in names:
        test_labels = 'test_label'
    else:
        test_labels = 'test_labels'


    x_train = train_file['train_data']
    x_test = test_file['test_data']
    y_train = train_file[train_labels]
    y_test = test_file[test_labels]

    

    x_train, y_train = shuffle(x_train, y_train, random_state=0)
    
   
    
    n_features = x_train.shape[-1]


    trainset = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    testset = torch.utils.data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shu, num_workers=worker, pin_memory=pin)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=worker, pin_memory=pin)



    return test_loader, train_loader, n_features, len(x_train)
   


def get_feature_data(feature_path, batch_size, shu, worker, pin):
    # get pre-computed features
    loaded_file = np.load(f"{feature_path}.npz")
    names = []
    for k in loaded_file.files:
        names.append(k)

    if "train_label" in names:
        train_labels = 'train_label'
    else:
        train_labels = 'train_labels'

    if "test_label" in names:
        test_labels = 'test_label'
    else:
        test_labels = 'test_labels'

    
    x_train = loaded_file['train_data']
    y_train = loaded_file[train_labels]
    x_test = loaded_file['test_data']
    y_test = loaded_file[test_labels]
   
 

    n_features = x_train.shape[-1]
    n_num = x_train.shape[0]
    
    #train_data, test_data = get_data("cifar10", augment=False)
    #y_train = np.asarray(train_data.targets)
    #y_test = np.asarray(test_data.targets)

    trainset = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    testset = torch.utils.data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shu, num_workers=worker, pin_memory=pin)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=worker, pin_memory=pin)

    return test_loader, train_loader, n_features, n_num


def get_data(name, augment=False, **kwargs):
    if name == "cifar100":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if augment:
            train_transforms = [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ]
        else:
            train_transforms = [
                transforms.ToTensor(),
                normalize,
            ]

        train_set = datasets.CIFAR100(root=".data", train=True,
                                     transform=transforms.Compose(train_transforms),
                                     download=True)

        test_set = datasets.CIFAR100(root=".data", train=False,
                                    transform=transforms.Compose(
                                        [transforms.ToTensor(), normalize]
                                    ))
    elif name == "cifar10":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if augment:
            train_transforms = [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ]
        else:
            train_transforms = [
                transforms.ToTensor(),
                normalize,
            ]

        train_set = datasets.CIFAR10(root=".data", train=True,
                                     transform=transforms.Compose(train_transforms),
                                     download=True)

        test_set = datasets.CIFAR10(root=".data", train=False,
                                    transform=transforms.Compose(
                                        [transforms.ToTensor(), normalize]
                                    ))

    elif name == "fmnist":
        train_set = datasets.FashionMNIST(root='.data', train=True,
                                          transform=transforms.ToTensor(),
                                          download=True)

        test_set = datasets.FashionMNIST(root='.data', train=False,
                                         transform=transforms.ToTensor(),
                                         download=True)

    elif name == "mnist":
        train_set = datasets.MNIST(root='.data', train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

        test_set = datasets.MNIST(root='.data', train=False,
                                  transform=transforms.ToTensor(),
                                  download=True)

    elif name == "cifar10_500K":

        # extended version of CIFAR-10 with pseudo-labelled tinyimages

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if augment:
            train_transforms = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]
        else:
            train_transforms = [
                transforms.ToTensor(),
                normalize,
            ]

        train_set = SemiSupervisedDataset(kwargs['aux_data_filename'],
                                          root=".data",
                                          train=True,
                                          download=True,
                                          transform=transforms.Compose(train_transforms))
        test_set = None
    else:
        raise ValueError(f"unknown dataset {name}")

    return train_set, test_set


class SemiSupervisedDataset(torch.utils.data.Dataset):
    def __init__(self,
                 aux_data_filename=None,
                 train=False,
                 **kwargs):
        """A dataset with auxiliary pseudo-labeled data"""

        self.dataset = datasets.CIFAR10(train=train, **kwargs)
        self.train = train

        # shuffle cifar-10
        p = np.random.permutation(len(self.data))
        self.data = self.data[p]
        self.targets = list(np.asarray(self.targets)[p])

        if self.train:
            self.sup_indices = list(range(len(self.targets)))
            self.unsup_indices = []

            aux_path = os.path.join(kwargs['root'], aux_data_filename)
            print("Loading data from %s" % aux_path)
            with open(aux_path, 'rb') as f:
                aux = pickle.load(f)
            aux_data = aux['data']
            aux_targets = aux['extrapolated_targets']
            orig_len = len(self.data)

            # shuffle additional data
            p = np.random.permutation(len(aux_data))
            aux_data = aux_data[p]
            aux_targets = aux_targets[p]

            self.data = np.concatenate((self.data, aux_data), axis=0)
            self.targets.extend(aux_targets)

            # note that we use unsup indices to track the labeled datapoints
            # whose labels are "fake"
            self.unsup_indices.extend(
                range(orig_len, orig_len+len(aux_data)))

            logger = logging.getLogger()
            logger.info("Training set")
            logger.info("Number of training samples: %d", len(self.targets))
            logger.info("Number of supervised samples: %d",
                        len(self.sup_indices))
            logger.info("Number of unsup samples: %d", len(self.unsup_indices))
            logger.info("Label (and pseudo-label) histogram: %s",
                        tuple(
                            zip(*np.unique(self.targets, return_counts=True))))
            logger.info("Shape of training data: %s", np.shape(self.data))

        # Test set
        else:
            self.sup_indices = list(range(len(self.targets)))
            self.unsup_indices = []

            logger = logging.getLogger()
            logger.info("Test set")
            logger.info("Number of samples: %d", len(self.targets))
            logger.info("Label histogram: %s",
                        tuple(
                            zip(*np.unique(self.targets, return_counts=True))))
            logger.info("Shape of data: %s", np.shape(self.data))

    @property
    def data(self):
        return self.dataset.data

    @data.setter
    def data(self, value):
        self.dataset.data = value

    @property
    def targets(self):
        return self.dataset.targets

    @targets.setter
    def targets(self, value):
        self.dataset.targets = value

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        self.dataset.labels = self.targets  # because torchvision is annoying
        return self.dataset[item]

    def __repr__(self):
        fmt_str = 'Semisupervised Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Training: {}\n'.format(self.train)
        fmt_str += '    Root Location: {}\n'.format(self.dataset.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.dataset.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.dataset.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class SemiSupervisedSampler(torch.utils.data.Sampler):
    def __init__(self, num_examples, num_batches, batch_size):
        self.inds = list(range(num_examples))
        self.batch_size = batch_size
        self.num_batches = num_batches
        super().__init__(None)

    def __iter__(self):
        batch_counter = 0
        inds_shuffled = [self.inds[i] for i in torch.randperm(len(self.inds))]

        while len(inds_shuffled) < self.num_batches*self.batch_size:
            temp = [self.inds[i] for i in torch.randperm(len(self.inds))]
            inds_shuffled.extend(temp)

        for k in range(0, self.num_batches*self.batch_size, self.batch_size):
            if batch_counter == self.num_batches:
                break

            batch = inds_shuffled[k:(k + self.batch_size)]

            # this shuffle operation is very important, without it
            # batch-norm / DataParallel hell ensues
            np.random.shuffle(batch)
            yield batch
            batch_counter += 1

    def __len__(self):
        return self.num_batches


class PoissonSampler(torch.utils.data.Sampler):
    def __init__(self, num_examples, batch_size):
        self.inds = np.arange(num_examples)
        self.batch_size = batch_size
        self.num_batches = int(np.ceil(num_examples / batch_size))
        self.sample_rate = self.batch_size / (1.0 * num_examples)
        super().__init__(None)

    def __iter__(self):
        # select each data point independently with probability `sample_rate`
        for i in range(self.num_batches):
            batch_idxs = np.random.binomial(n=1, p=self.sample_rate, size=len(self.inds))
            batch = self.inds[batch_idxs.astype(np.bool)]
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches





































# import os
# import pickle
# import numpy as np
# import logging

# import torch
# from torchvision import datasets, transforms


# SHAPES = {
#     "cifar10": (32, 32, 3),
#     "cifar10_500K": (32, 32, 3),
#     "fmnist": (28, 28, 1),
#     "mnist": (28, 28, 1)
# }

# def get_feature_data(dataset, feature_path, batch_size, shu, worker, pin):
#     # get pre-computed features
#     loaded_file = np.load(f"{feature_path}.npz")
#     x_train = loaded_file['train_data']
#     x_test = loaded_file['test_data']
#     y_train = loaded_file['train_labels']
#     y_test = loaded_file['test_labels']

#     n_features = x_train.shape[-1]
    
#     # train_data, test_data = get_data(dataset, augment=False)
#     # y_train = np.asarray(train_data.targets)
#     # y_test = np.asarray(test_data.targets)

#     trainset = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
#     testset = torch.utils.data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
#     train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shu, num_workers=worker, pin_memory=pin)
#     test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=worker, pin_memory=pin)

#     return test_loader, train_loader, n_features, len(x_train)


# def get_data(name, augment=False, **kwargs):
#     if name == "cifar100":
#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                          std=[0.229, 0.224, 0.225])

#         if augment:
#             train_transforms = [
#                     transforms.RandomHorizontalFlip(),
#                     transforms.RandomCrop(32, 4),
#                     transforms.ToTensor(),
#                     normalize,
#                 ]
#         else:
#             train_transforms = [
#                 transforms.ToTensor(),
#                 normalize,
#             ]

#         train_set = datasets.CIFAR100(root=".data", train=True,
#                                      transform=transforms.Compose(train_transforms),
#                                      download=True)

#         test_set = datasets.CIFAR100(root=".data", train=False,
#                                     transform=transforms.Compose(
#                                         [transforms.ToTensor(), normalize]
#                                     ))
#     elif name == "cifar10":
#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                          std=[0.229, 0.224, 0.225])

#         if augment:
#             train_transforms = [
#                     transforms.RandomHorizontalFlip(),
#                     transforms.RandomCrop(32, 4),
#                     transforms.ToTensor(),
#                     normalize,
#                 ]
#         else:
#             train_transforms = [
#                 transforms.ToTensor(),
#                 normalize,
#             ]

#         train_set = datasets.CIFAR10(root=".data", train=True,
#                                      transform=transforms.Compose(train_transforms),
#                                      download=True)

#         test_set = datasets.CIFAR10(root=".data", train=False,
#                                     transform=transforms.Compose(
#                                         [transforms.ToTensor(), normalize]
#                                     ))

#     elif name == "fmnist":
#         train_set = datasets.FashionMNIST(root='.data', train=True,
#                                           transform=transforms.ToTensor(),
#                                           download=True)

#         test_set = datasets.FashionMNIST(root='.data', train=False,
#                                          transform=transforms.ToTensor(),
#                                          download=True)

#     elif name == "mnist":
#         train_set = datasets.MNIST(root='.data', train=True,
#                                    transform=transforms.ToTensor(),
#                                    download=True)

#         test_set = datasets.MNIST(root='.data', train=False,
#                                   transform=transforms.ToTensor(),
#                                   download=True)

#     elif name == "cifar10_500K":

#         # extended version of CIFAR-10 with pseudo-labelled tinyimages

#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                          std=[0.229, 0.224, 0.225])

#         if augment:
#             train_transforms = [
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomCrop(32, 4),
#                 transforms.ToTensor(),
#                 normalize,
#             ]
#         else:
#             train_transforms = [
#                 transforms.ToTensor(),
#                 normalize,
#             ]

#         train_set = SemiSupervisedDataset(kwargs['aux_data_filename'],
#                                           root=".data",
#                                           train=True,
#                                           download=True,
#                                           transform=transforms.Compose(train_transforms))
#         test_set = None
#     else:
#         raise ValueError(f"unknown dataset {name}")

#     return train_set, test_set


# class SemiSupervisedDataset(torch.utils.data.Dataset):
#     def __init__(self,
#                  aux_data_filename=None,
#                  train=False,
#                  **kwargs):
#         """A dataset with auxiliary pseudo-labeled data"""

#         self.dataset = datasets.CIFAR10(train=train, **kwargs)
#         self.train = train

#         # shuffle cifar-10
#         p = np.random.permutation(len(self.data))
#         self.data = self.data[p]
#         self.targets = list(np.asarray(self.targets)[p])

#         if self.train:
#             self.sup_indices = list(range(len(self.targets)))
#             self.unsup_indices = []

#             aux_path = os.path.join(kwargs['root'], aux_data_filename)
#             print("Loading data from %s" % aux_path)
#             with open(aux_path, 'rb') as f:
#                 aux = pickle.load(f)
#             aux_data = aux['data']
#             aux_targets = aux['extrapolated_targets']
#             orig_len = len(self.data)

#             # shuffle additional data
#             p = np.random.permutation(len(aux_data))
#             aux_data = aux_data[p]
#             aux_targets = aux_targets[p]

#             self.data = np.concatenate((self.data, aux_data), axis=0)
#             self.targets.extend(aux_targets)

#             # note that we use unsup indices to track the labeled datapoints
#             # whose labels are "fake"
#             self.unsup_indices.extend(
#                 range(orig_len, orig_len+len(aux_data)))

#             logger = logging.getLogger()
#             logger.info("Training set")
#             logger.info("Number of training samples: %d", len(self.targets))
#             logger.info("Number of supervised samples: %d",
#                         len(self.sup_indices))
#             logger.info("Number of unsup samples: %d", len(self.unsup_indices))
#             logger.info("Label (and pseudo-label) histogram: %s",
#                         tuple(
#                             zip(*np.unique(self.targets, return_counts=True))))
#             logger.info("Shape of training data: %s", np.shape(self.data))

#         # Test set
#         else:
#             self.sup_indices = list(range(len(self.targets)))
#             self.unsup_indices = []

#             logger = logging.getLogger()
#             logger.info("Test set")
#             logger.info("Number of samples: %d", len(self.targets))
#             logger.info("Label histogram: %s",
#                         tuple(
#                             zip(*np.unique(self.targets, return_counts=True))))
#             logger.info("Shape of data: %s", np.shape(self.data))

#     @property
#     def data(self):
#         return self.dataset.data

#     @data.setter
#     def data(self, value):
#         self.dataset.data = value

#     @property
#     def targets(self):
#         return self.dataset.targets

#     @targets.setter
#     def targets(self, value):
#         self.dataset.targets = value

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, item):
#         self.dataset.labels = self.targets  # because torchvision is annoying
#         return self.dataset[item]

#     def __repr__(self):
#         fmt_str = 'Semisupervised Dataset ' + self.__class__.__name__ + '\n'
#         fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
#         fmt_str += '    Training: {}\n'.format(self.train)
#         fmt_str += '    Root Location: {}\n'.format(self.dataset.root)
#         tmp = '    Transforms (if any): '
#         fmt_str += '{0}{1}\n'.format(tmp, self.dataset.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
#         tmp = '    Target Transforms (if any): '
#         fmt_str += '{0}{1}'.format(tmp, self.dataset.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
#         return fmt_str


# class SemiSupervisedSampler(torch.utils.data.Sampler):
#     def __init__(self, num_examples, num_batches, batch_size):
#         self.inds = list(range(num_examples))
#         self.batch_size = batch_size
#         self.num_batches = num_batches
#         super().__init__(None)

#     def __iter__(self):
#         batch_counter = 0
#         inds_shuffled = [self.inds[i] for i in torch.randperm(len(self.inds))]

#         while len(inds_shuffled) < self.num_batches*self.batch_size:
#             temp = [self.inds[i] for i in torch.randperm(len(self.inds))]
#             inds_shuffled.extend(temp)

#         for k in range(0, self.num_batches*self.batch_size, self.batch_size):
#             if batch_counter == self.num_batches:
#                 break

#             batch = inds_shuffled[k:(k + self.batch_size)]

#             # this shuffle operation is very important, without it
#             # batch-norm / DataParallel hell ensues
#             np.random.shuffle(batch)
#             yield batch
#             batch_counter += 1

#     def __len__(self):
#         return self.num_batches


# class PoissonSampler(torch.utils.data.Sampler):
#     def __init__(self, num_examples, batch_size):
#         self.inds = np.arange(num_examples)
#         self.batch_size = batch_size
#         self.num_batches = int(np.ceil(num_examples / batch_size))
#         self.sample_rate = self.batch_size / (1.0 * num_examples)
#         super().__init__(None)

#     def __iter__(self):
#         # select each data point independently with probability `sample_rate`
#         for i in range(self.num_batches):
#             batch_idxs = np.random.binomial(n=1, p=self.sample_rate, size=len(self.inds))
#             batch = self.inds[batch_idxs.astype(np.bool)]
#             np.random.shuffle(batch)
#             yield batch

#     def __len__(self):
#         return self.num_batches