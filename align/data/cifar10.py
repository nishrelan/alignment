from haiku import Flatten
import torch
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import jax
import jax.numpy as jnp
import sys


def collate_fn(samples):
    xb, yb = list(zip(*samples))
    xb = np.stack(xb)
    yb = np.array(yb)
    return xb, yb

class ToNumpyTransform:
    def __init__(self) -> None:
        pass
    def __call__(self, x):
        return x

class FlattenTransform:
    def __init__(self) -> None:
        pass
    def __call__(self, x):
        return x.reshape(-1)

def get_cifar10(path, batch_size, train_size=None, test_size=None, rngs=None):
    """
    Params:
        batch_size: batch size for both train and test loaders
        train_size: number of data points to include in training set
        test_size: number of data points to include in test set
        rngs: rng keys needed to select random subsets of data of size train_size and test_size
    Returns:
        train and test data loaders 
    
    """
    train_ds = torchvision.datasets.CIFAR10(path, train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                ToNumpyTransform(),
                                ]),
                                target_transform=torchvision.transforms.Compose([
                                    ToNumpyTransform(),
                                ]))
    test_ds = torchvision.datasets.CIFAR10(path, train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                ToNumpyTransform(),
                                ]),
                                target_transform=torchvision.transforms.Compose([
                                    ToNumpyTransform(),
                                ]))

    # select random subset of train and test dataset   
    trainloader = DataLoader(train_ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    testloader = DataLoader(test_ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    return trainloader, testloader