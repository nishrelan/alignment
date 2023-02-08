import neural_tangents as nt
import jax
import jax.numpy as jnp
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd
import os
import sys
import logging
from jax.random import split


from align.utils.comp import print_tree
from align.utils.config import get_optimizer
from align.data.cifar10 import get_cifar10
from align.models.cnn import create_model
from align.train_with_state import *

log = logging.getLogger(__name__)



@hydra.main(version_base=None, config_path="./config/cnn_config", config_name="config")
def main(config):
    seed = jax.random.PRNGKey(config.seed)
    key1, key2, key3, key4, key5, splitter = split(seed, num=6)
    train_loader, test_loader = get_cifar10(
        os.path.join(get_original_cwd(), 'datasets'), batch_size=config.data.batch_size)
    xb, yb = next(iter(train_loader))


    model, init_params, init_state = create_model('vgg11', config.architecture, key1, xb)
    optimizer = get_optimizer(config.optimizer.type, **config.optimizer.spec)
    init_opt_state = optimizer.init(init_params)
    xent_loss_fn, acc_fn = get_xent_loss_acc(model.apply)
    train_step_fn = get_train_step_fn(optimizer, xent_loss_fn)


    log.info("Training VGG11 on CIFAR10...")
    results, _, _, _ = train(
        init_params, init_state, init_opt_state, train_step_fn,
        train_loader, test_loader, xent_loss_fn, acc_fn,
        num_iters=config.epochs
    )

    
    







if __name__ == '__main__':
    main()



