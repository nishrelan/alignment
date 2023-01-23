import jax
from jax.tree_util import tree_map
import jax.numpy as jnp
import optax


def tuple_split(thing):
    return map(list, zip(*thing))

def print_tree(pytree):
    print(
        tree_map(lambda x: x.shape, pytree)
    )

def random_labels(key, num_points=100):
    return jax.random.choice(key, jnp.array([-1, 1]), shape=(num_points,))



# TODO: Figure out what to do with this
def avg_first_layer_magnitude(*paramss):
    res = []
    for params in paramss:
        res.append(jnp.mean(params['linear']['w'], axis=1))
    return res