import jax
from jax.tree_util import tree_map
import jax.numpy as jnp
import optax
from align.models.mlp import *


def get_activation(act_str):
    if act_str == "relu":
        return jax.nn.relu
    elif act_str == "sigmoid":
        return jax.nn.sigmoid
    elif act_str == "selu":
        return jax.nn.selu
    elif act_str == "tanh":
        return jax.nn.tanh
    else:
        raise NotImplementedError

def get_optimizer(opt_str, **kwargs):
    if opt_str == "sgd":
        return optax.sgd(**kwargs)
    else:
        raise NotImplementedError

def get_model_create_fn(model_str):
    if model_str == 'mlp':
        return create_mlp
    elif model_str == 'paper':
        return create_two_layer


def get_model_and_optimizer(config, name, rng, sample_data):
    if name == 'paper':
        model, init_params = create_two_layer(
            rng, sample_data, config.hidden_nodes, get_activation(config.activation)
            )
        optimizer = get_optimizer(config.optimizer.type, **config.optimizer.spec)
        init_opt_state = optimizer.init(init_params)
        return model, init_params, optimizer, init_opt_state
    elif name == 'mlp':
        model, init_params = create_mlp(
            rng, sample_data, config.layers, get_activation(config.activation)
        )
        optimizer = get_optimizer(config.optimizer.type, **config.optimizer.spec)
        init_opt_state = optimizer.init(init_params)
        return model, init_params, optimizer, init_opt_state
    else:
        raise NotImplementedError

        
