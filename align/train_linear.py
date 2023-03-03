from align.models.mlp import create_two_layer
from align.data.stripe import generate_data
from align.train import get_hinge_loss, get_update_fun, make_acc_fn, train
from utils.comp import random_labels, tuple_split, print_tree
from utils.config import *
from align.alignment_metrics import energy_concentration_fun, get_ntk_alignment_fn, get_normed_alignment_fn

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
import os
import logging
import time

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./config/mlp_config", config_name="config")
def print_config(cfg):
    print(OmegaConf.to_yaml(cfg['model']))


@hydra.main(version_base=None, config_path="./config/mlp_config", config_name="config")
def main(config):
    seed = jax.random.PRNGKey(config.rng_seed)
    key1, key2, key3, key4, key5, splitter = split(seed, num=6)

    conf = config.data
    train_loader, test_loader = generate_data(key1, conf.num_train, conf.num_test, conf.dim)
    x, y = next(iter(train_loader))
    x_test, y_test = next(iter(test_loader))


    # Full paper model
    model_name = config.model_name
    model, init_params, optimizer = get_model_and_optimizer(
        config.model[model_name], model_name, key2, x
    )

    @jax.jit
    def f(params, x):
        return model.apply(params, x) - model.apply(init_params, x)


    ntk = nt.batch(
        nt.empirical_kernel_fn(f, vmap_axes=0),
        device_count=0
    )

    g_dd = ntk(x, None, 'ntk', init_params)
    g_td = ntk(x_test, x, 'ntk', init_params)

    @jax.jit
    def hinge_loss(fx, y):
        fx = jnp.ravel(fx)
        y = jnp.ravel(y)
        preds = config.alpha*fx
        preds = 1 - y*preds
        res = jnp.mean(jnp.where(preds < 0, 0, preds)) / config.alpha
        return res

    mse_loss = jax.jit(lambda fx, y_hat: 0.5 * np.mean((fx - y_hat) ** 2))


    predictor = nt.predict.gradient_descent(hinge_loss, g_dd, jnp.expand_dims(y, axis=1), learning_rate=50000, momentum=0.9)
    fx_train = f(init_params, x)
    training_steps = np.linspace(0, 1000, 10)
    print("Doing function space gradient descent...")
    start = time.time()
    predictions = predictor(training_steps, fx_train)
    print("Time: {}".format(time.time() - start))
    print(hinge_loss(predictions[-1], jnp.expand_dims(y, axis=1)))


    



if __name__ == '__main__':
    main()