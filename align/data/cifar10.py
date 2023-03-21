from haiku import Flatten
import torch
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import jax
import jax.numpy as jnp
import sys
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from scipy.fftpack import dct, idct
from skimage.color import rgb2gray



def addHFNoise(x, mfilter, multiplier):
    """This function adds Gaussian noise to frequencies identified by mfilter.
    Image cannot have channel dimension, ie this function assumes images are in
    greyscale
    Inputs:
        X: numpy array of size d x d. 
        mfilter: A matrix of {0, 1} of size d x d
        multiplier: A positive constant corresponding to noise to signal ratio.
    Outputs:
        Z: Noisy image."""
    d = len(x)
    z = np.zeros(d)
    image = x + 0.0
    image = image - np.mean(image)		
    freq = dct(dct(image, axis=0, norm='ortho'), axis=1, norm='ortho')
    noise = np.random.normal(size=(d, d))
    noise = np.multiply(noise, mfilter)
    noise = noise / np.linalg.norm(noise)
    noise = noise * (multiplier * np.linalg.norm(freq))
    noisy_freq = freq + noise
    newIm = idct(idct(noisy_freq, axis=0, norm='ortho'), axis=1, norm='ortho') 
    newIm = newIm / np.linalg.norm(newIm)
    newIm = newIm.flatten() * d
    return newIm


def collate_fn(samples):
    xb, yb = list(zip(*samples))
    xb = np.stack(xb)
    yb = np.array(yb)
    return xb, yb

class NormalizeTransform:
    def __init__(self) -> None:
        pass
    def __call__(self, x):
        d = 32
        x = x.reshape(d, d, 3)
        x = rgb2gray(x)
        x = x - np.mean(x)
        x = x / np.linalg.norm(x.flatten()) * d
        return x


class HFNTransform:
    def __init__(self, noise_ratio):
        self.noise_ratio = noise_ratio
        dim = 32
        self.cfilter = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                radius = (dim - i - 1) ** 2 + (dim - j - 1) ** 2
                radius = np.sqrt(radius)
                if radius <= (dim - 1):
                    self.cfilter[i, j] = 1.0
    
    def __call__(self, x):
        return addHFNoise(x, self.cfilter, self.noise_ratio)

class FlattenTransform:
    def __init__(self) -> None:
        self.called = False
    def __call__(self, x):
        return x.reshape(-1)
    


def get_cifar_2(path, batch_size, hf_noise_ratio, flatten=True):
    """
    Params:
        batch_size: batch size for both train and test loaders
        train_size: number of data points to include in training set
        test_size: number of data points to include in test set
        rngs: rng keys needed to select random subsets of data of size train_size and test_size
    Returns:
        train and test data loaders 
    
    """
    transforms = [
        torchvision.transforms.ToTensor(),
        NormalizeTransform(),
        HFNTransform(noise_ratio=hf_noise_ratio)
    ]
    if flatten:
        transforms.append(FlattenTransform())

    train_ds = torchvision.datasets.CIFAR10(path, train=True, download=True,
                                transform=torchvision.transforms.Compose(transforms),
                                )
    test_ds = torchvision.datasets.CIFAR10(path, train=False, download=True,
                                transform=torchvision.transforms.Compose(transforms),
                                )
  

    train_targets = np.array(train_ds.targets)
    test_targets = np.array(test_ds.targets)
    train_mask = np.logical_or(train_targets == 0, train_targets == 3)
    test_mask = np.logical_or(test_targets == 0, test_targets == 3)
    train_idxs = np.ravel(np.argwhere(train_mask == True))
    test_idxs = np.ravel(np.argwhere(test_mask == True))

    train_targets = train_targets[train_idxs]
    test_targets = test_targets[test_idxs]
    train_ds.data = train_ds.data[train_idxs]
    test_ds.data = test_ds.data[test_idxs]

    train_ds.targets = np.where(np.array(train_targets) == 3, 1, 0)
    test_ds.targets = np.where(np.array(test_targets) == 3, 1, 0)

    trainloader = DataLoader(train_ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    testloader = DataLoader(test_ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    return trainloader, testloader