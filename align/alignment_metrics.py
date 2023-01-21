import jax
import jax.numpy as jnp
import optax
import sys
import neural_tangents as nt
from scipy.sparse.linalg import eigsh
import numpy as np

from functools import partial
import sys
from align.utils.comp import print_tree




def energy_concentration_fun(model, batch, params, k):
    def get_ntk_fn(apply_fun):
        ntk_fn = nt.empirical_kernel_fn(apply_fun, vmap_axes=0)
        return ntk_fn

    def eigen_decomposition(ntk_fn, data, params, k=50):
        graham_matrix = ntk_fn(data, None, "ntk", params)
        eigvals, eigvecs = eigsh(jax.device_get(graham_matrix), k=k)
        eigvals = np.flipud(eigvals)
        eigvecs = np.flipud(eigvecs.T)
        return eigvals, eigvecs

    def energy_concentration(eigvecs, labels):
        return jnp.linalg.norm(eigvecs @ labels) / jnp.linalg.norm(labels)

    
    xb, yb = batch
    ntk_fn = get_ntk_fn(model.apply)
    _, eigvecs = eigen_decomposition(ntk_fn, xb, params, k)
    return energy_concentration(eigvecs, yb)



def get_ntk_alignment_fn(model):
    
    def compute_alignment(batch, params):
        xb, yb = batch
        jac = jax.jacrev(partial(model.apply, x=xb))(params)
        assert(len(yb.shape) == 1)
        # 1. scale each gradient and sum along data axis
        scaled = jax.tree_map(lambda x: jnp.einsum('i, ij... -> ...', yb, x) / len(yb), jac)
        # 2. compute norm
        norm = jnp.sum(jnp.array(
            [jnp.sum(l**2) for l in jax.tree_leaves(scaled)]
        ))
        return norm



    return compute_alignment






