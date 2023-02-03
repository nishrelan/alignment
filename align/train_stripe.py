from align.models.mlp import create_two_layer
from align.data.polynomial import generate_data
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
    model_name = config.model_name
    model, init_params, optimizer = get_model_and_optimizer(
        config.model[model_name], model_name, key2, x
    )
    init_opt_state, loss_fn, acc_fn, train_step_fn = setup_training(
        config.loss, config.alpha, model.apply, init_params, optimizer
    )

    # Approximations
    lin = taylor_expand(model.apply, init_params, degree=1)
    lin_init_opt_state, lin_loss_fn, lin_acc_fn, lin_train_step_fn = setup_training(
        config.loss, config.alpha, lin, init_params, optimizer
    )

    quad = taylor_expand(model.apply, init_params, degree=2)
    quad_init_opt_state, quad_loss_fn, quad_acc_fn, quad_train_step_fn = setup_training(
        config.loss, config.alpha, quad, init_params, optimizer
    )




    metrics = [
        Metric('energy', partial(energy_concentration_fun, model=model.apply, k=8), config.metrics.energy_concentration),
        Metric('normed_alignment', get_normed_alignment_fn(model.apply), config.metrics.normed_alignment),
        Metric('test_loss', loss_fn, 1)
    ]

    lin_metrics = [
        Metric('energy', partial(energy_concentration_fun, model=lin, k=8), config.metrics.energy_concentration),
        Metric('normed_alignment', get_normed_alignment_fn(lin), config.metrics.normed_alignment),
        Metric('test_loss', lin_loss_fn, 1)
    ]

    quad_metrics = [
        Metric('energy', partial(energy_concentration_fun, model=quad, k=8), config.metrics.energy_concentration),
        Metric('normed_alignment', get_normed_alignment_fn(quad), config.metrics.normed_alignment),
        Metric('test_loss', quad_loss_fn, 1)
    ]



    print("Training full model...")
    results, _, _ = train(
        init_params, init_opt_state, train_step_fn, train_loader, 
        test_loader, acc_fn, num_epochs=config.epochs, metrics=metrics
    )

    print("Training linear approximation...")
    lin_results, _, _ = train(
        init_params, lin_init_opt_state, lin_train_step_fn, train_loader,
        test_loader, lin_acc_fn, config.epochs, lin_metrics
    )

    print("Training quadratic approximation")
    quad_results, _, _ = train(
        init_params, quad_init_opt_state, quad_train_step_fn,
        train_loader, test_loader, quad_acc_fn,
        config.epochs, quad_metrics
    )

    

    
    losses, epochs = tuple_split(results['test_loss'])
    lin_losses, lin_epochs = tuple_split(lin_results['test_loss'])
    quad_losses, quad_epochs = tuple_split(quad_results['test_loss'])
    plt.clf()
    plt.plot(epochs, losses, label='full model')
    plt.plot(lin_epochs, lin_losses, label='lin')
    plt.plot(quad_epochs, quad_losses, label='quad')
    plt.legend()
    plt.savefig('test_loss.png')

    accs, epochs = tuple_split(results['test_acc'])
    lin_accs, lin_epochs = tuple_split(lin_results['test_acc'])
    quad_accs, quad_epochs = tuple_split(quad_results['test_acc'])
    plt.clf()
    plt.plot(epochs, accs, label='full model')
    plt.plot(lin_epochs, lin_accs, label='lin')
    plt.plot(quad_epochs, quad_accs, label='quad')
    plt.legend()
    plt.savefig('test_acc.png')

    
    # alignments, epochs = tuple_split(results['normed_alignment'])
    # lin_alignments, lin_epochs = tuple_split(lin_results['normed_alignment'])
    # quad_alignments, quad_epochs = tuple_split(quad_results['normed_alignment'])
    # plt.clf()
    # plt.plot(epochs, alignments, label='full model')
    # plt.plot(lin_epochs, lin_alignments, label='lin')
    # plt.plot(quad_epochs, quad_alignments, label='quad')
    # plt.legend()
    # plt.savefig('alignment_curve.png')

    # energies, epochs = tuple_split(results['energy'])
    # lin_energies, lin_epochs = tuple_split(lin_results['energy'])
    # quad_energies, quad_epochs = tuple_split(quad_results['energy'])
    # plt.clf()
    # plt.plot(epochs, energies, label="full model")
    # plt.plot(lin_epochs, lin_energies, label="lin")
    # plt.plot(quad_epochs, quad_energies, label='quad')
    # plt.legend()
    # plt.savefig('energy (k={})'.format(8))







if __name__ == '__main__':
    main()