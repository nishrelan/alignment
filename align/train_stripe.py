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

from omegaconf import DictConfig, OmegaConf
import hydra
import os


class Metric:
    def __init__(self, name, metric_fn, interval) -> None:
        self.name = name
        self.fun  = metric_fn
        self.interval = interval

@hydra.main(version_base=None, config_path="./config", config_name="config")
def print_config(cfg):
    print(OmegaConf.to_yaml(cfg['model']))


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    seed = jax.random.PRNGKey(1)
    key1, key2, key3, key4, key5, splitter = split(seed, num=6)

    conf = config.data
    train_loader, test_loader = generate_data(key1, conf.num_train, conf.num_test, conf.dim)
    x, y = next(iter(train_loader))
    print(len(x))


    # Full paper model
    model_name = 'mlp'
    model, init_params, optimizer, init_opt_state = get_model_and_optimizer(
        config.model[model_name], model_name, key2, x
    )
    hinge_loss = get_hinge_loss(model.apply, init_params, alpha=config.alpha)
    acc_fn = make_acc_fn(model.apply, init_params, alpha=config.alpha)
    train_step_fn = get_update_fun(optimizer, hinge_loss)

    # Approximation
    lin = taylor_expand(model.apply, init_params, degree=2)
    lin_hinge_loss = get_hinge_loss(lin, init_params, alpha=config.alpha)
    lin_acc_fn = make_acc_fn(lin, init_params, alpha=config.alpha)
    lin_init_opt_state = optimizer.init(init_params)
    lin_train_step_fn = get_update_fun(optimizer, lin_hinge_loss)


    metrics = [
        Metric('energy', partial(energy_concentration_fun, model=model.apply, k=8), config.metrics.energy_concentration),
        Metric('normed_alignment', get_normed_alignment_fn(model.apply), config.metrics.normed_alignment)
    ]

    lin_metrics = [
        Metric('energy', partial(energy_concentration_fun, model=lin, k=8), config.metrics.energy_concentration),
        Metric('normed_alignment', get_normed_alignment_fn(lin), config.metrics.normed_alignment)
    ]


    results, final_params, final_opt_state = train(
        init_params, init_opt_state, train_step_fn, train_loader, 
        test_loader, acc_fn, num_epochs=config.epochs, metrics=metrics
    )

    lin_results, _, _ = train(
        init_params, lin_init_opt_state, lin_train_step_fn, train_loader,
        test_loader, lin_acc_fn, config.epochs, lin_metrics
    )

    
    accs, epochs = tuple_split(results['test_acc'])
    lin_accs, lin_epochs = tuple_split(lin_results['test_acc'])

    plt.plot(epochs, accs, label='full model')
    plt.plot(lin_epochs, lin_accs, label='quad')
    plt.legend()
    plt.savefig('acc_curve1.png')

    plt.clf()
    energies, epochs = tuple_split(results['normed_alignment'])
    lin_energies, lin_epochs = tuple_split(lin_results['normed_alignment'])
    plt.plot(epochs, energies, label='full_model')
    plt.plot(lin_epochs, lin_energies, label='quad_model')
    plt.legend()
    plt.savefig('alignment_curve.png')







if __name__ == '__main__':
    main()