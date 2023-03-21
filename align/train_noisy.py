from align.models.mlp import create_two_layer, create_mlp
from align.data.stripe import generate_data
from align.data.cifar10 import get_cifar_2
from align.batch_train import get_bce_loss_acc, get_mse_loss_acc, train
from utils.comp import random_labels, tuple_split, print_tree



import jax
import jax.numpy as jnp
from jax.random import split
import optax
import numpy as np
import sys
from neural_tangents import taylor_expand
import matplotlib.pyplot as plt
from inspect import signature
import neural_tangents as nt

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import get_original_cwd
import os
import logging
import time

log = logging.getLogger(__name__)
from neural_tangents import stax


@hydra.main(version_base=None, config_path="./config/noisy_config", config_name="config")
def print_config(cfg):
    print(OmegaConf.to_yaml(cfg['model']))


@hydra.main(version_base=None, config_path="./config/noisy_config", config_name="config")
def main(config):
    num_points = 10000
    seed = jax.random.PRNGKey(2)
    key1, key2, key3, key4, key5, splitter = split(seed, num=6)

    data_path = os.path.join(get_original_cwd(), 'datasets')
    train_loader, test_loader = get_cifar_2(path=data_path, batch_size=config.batch_size, hf_noise_ratio=config.tau)

    x, y = next(iter(train_loader))
    x_test, y_test = next(iter(test_loader))
    model, init_params = create_mlp(key1, sample_data=x, output_sizes=[4096], act_fn=jax.nn.relu)
    lin = taylor_expand(model.apply, init_params, 1)
    quad = taylor_expand(model.apply, init_params, 2)
    optimizer = optax.sgd(5e-4, momentum=0.9)
    init_opt_state = optimizer.init(init_params)
    
    if config.approx == 0:
        loss_fn, acc_fn = get_mse_loss_acc(model.apply)
    elif config.approx == 1:
        print("LINEAR")
        loss_fn, acc_fn = get_mse_loss_acc(lin)
    elif config.approx == 2:
        loss_fn, acc_fn = get_mse_loss_acc(quad)
    else:
        raise ValueError

    
    loss_fn(init_params, (x,y))

    results, final_params, opt_state = train(
        init_params, optimizer, init_opt_state, train_loader, test_loader,
        loss_fn, acc_fn, num_iters=int(num_points/config.batch_size*config.num_epochs), print_iter=20
    )


    



if __name__ == '__main__':
    main()
