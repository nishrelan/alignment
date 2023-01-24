import haiku as hk
import jax
import jax.numpy as jnp
from functools import partial
import sys


class MLP(hk.Module):
    def __init__(self, output_sizes, activation_fn):
        super().__init__()
        self.output_sizes = output_sizes + [1]
        self.mlp = hk.nets.MLP(output_sizes=self.output_sizes, activation=activation_fn)

    def __call__(self, x):
        return self.mlp(x)


class TwoLayer(hk.Module):
    def __init__(self, hidden_size, activation_fn):
        super().__init__()
        self.linear1 = hk.Linear(
            output_size=hidden_size, with_bias=False, 
            w_init=hk.initializers.RandomNormal())
        self.linear2 = hk.Linear(output_size=1, with_bias=False, w_init=hk.initializers.RandomNormal())
        self.b_init = hk.initializers.RandomNormal()
        self.hidden_size = hidden_size
        self.activation = activation_fn

    def __call__(self, x):
        n, d = x.shape
        b = hk.get_parameter('bias', shape=(self.hidden_size,), init=self.b_init)
        x = self.linear1(x)
        x = x / jnp.sqrt(d) + b
        return self.linear2(jnp.sqrt(2)*self.activation(x)) / self.hidden_size


def create_two_layer(rng_key, sample_data, hidden_size, act_fn):
    def forward(x, h, a):
        model = TwoLayer(hidden_size=h, activation_fn=a)
        return model(x)
    model = hk.without_apply_rng(hk.transform(partial(forward, h=hidden_size, a=act_fn)))
    init_params = model.init(rng_key, sample_data)
    return model, init_params

def create_mlp(rng_key, sample_data, output_sizes, act_fn):
    def forward(x, o, a):
        model = MLP(output_sizes=o, activation_fn=a)
        return model(x)
    model = hk.without_apply_rng(hk.transform(partial(forward, o=output_sizes, a=act_fn)))
    init_params = model.init(rng_key, sample_data)
    return model, init_params
