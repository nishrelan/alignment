import jax
import jax.numpy as jnp
from functools import partial
from jax.lax import cond
import torch
from torch.utils.data import Dataset, DataLoader


class NumpyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.data)

def collate_fn(samples):
    xb, yb = list(zip(*samples))
    xb = jnp.stack(xb)
    yb = jnp.array(yb)
    return xb, yb


@partial(jax.vmap, in_axes=0)
def decision_func(point):
    return cond(jax.lax.bitwise_or(point[0] < -0.3, point[0] > 1.18549), lambda: 1, lambda: -1)

def generate_data(rng, num_train, num_test, d):
    data = jax.random.normal(rng, shape=(num_train + num_test, d))
    labels = decision_func(data)
    xtrain = data[:num_train]
    ytrain = labels[:num_train]
    xtest = data[num_train:]
    ytest = labels[num_train:]

    return (
        DataLoader(NumpyDataset(xtrain, ytrain), batch_size=len(xtrain), shuffle=True, collate_fn=collate_fn),
        DataLoader(NumpyDataset(xtest, ytest), batch_size=len(xtest), shuffle=False, collate_fn=collate_fn)
    )
    
    