from align.models.mlp import create_two_layer
from align.data.stripe import generate_data
from align.train import get_hinge_loss, get_update_fun, make_acc_fn, train
from utils.comp import random_labels
from utils.config import *
from align.alignment_metrics import energy_concentration_fun, get_ntk_alignment_fn

import jax
import jax.numpy as jnp
from jax.random import split
import optax
import numpy as np
import sys

from omegaconf import DictConfig, OmegaConf
import hydra


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



    model_name = 'paper'
    model, init_params, optimizer, init_opt_state = get_model_and_optimizer(
        config.model[model_name], model_name, key2, x
    )
    hinge_loss = get_hinge_loss(model, init_params, alpha=config.alpha)
    acc_fn = make_acc_fn(model, init_params, alpha=config.alpha)
    train_step_fn = get_update_fun(optimizer, hinge_loss)

    metrics = [
        Metric('energy', partial(energy_concentration_fun, model=model, k=8), config.metrics.energy_concentration),
        Metric('alignment', get_ntk_alignment_fn(model), config.metrics.target_alignment),
        Metric('acc', acc_fn, config.metrics.curves)
    ]


    results, final_params, final_opt_state = train(
        init_params, init_opt_state, train_step_fn, train_loader, 
        test_loader, acc_fn, num_epochs=config.epochs, metrics=metrics
    )
    print(results)






if __name__ == '__main__':
    main()


