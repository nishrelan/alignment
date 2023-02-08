import jax
import jax.numpy as jnp
import optax
import sys
from functools import partial
import logging

log = logging.getLogger(__name__)



def train(params, state, opt_state, train_step_fn, train_loader, test_loader, loss_fn, acc_fn, num_iters):
    results = {
        'train_acc': [],
        'test_acc': [],
        'train_loss': [],
        'test_loss': []
    }
    def get_next_batch(iterable_loader, loader):
        try:
            batch = next(iterable_loader)
        except StopIteration:
            iterable_loader = iter(loader)
            batch = next(iterable_loader)
        return iterable_loader, batch

    train_iter = iter(train_loader)
    test_iter = iter(test_loader)

    for i in range(num_iters):
        train_iter, train_batch = get_next_batch(train_iter, train_loader)
        test_iter, test_batch = get_next_batch(test_iter, test_loader)


        train_acc = acc_fn(params, state, train_batch)
        test_acc = acc_fn(params, state, test_batch)
        test_loss, _ = loss_fn(params, state, test_batch, is_training=False)
        train_loss, state, params, opt_state = train_step_fn(params, state, train_batch, opt_state)

        log.info(
            "Epoch: {} Train Loss: {} Test Loss: {} Train Acc: {} Test Acc: {}".format(
                i, train_loss, test_loss, train_acc, test_acc
            )
        )
        results['train_acc'].append(train_acc)
        results['test_acc'].append(test_acc)
        results['test_loss'].append(test_loss)
        results['train_loss'].append(train_loss)
    
    return results, params, state, opt_state


            





def get_xent_loss_acc(apply_fn):

    @partial(jax.jit, static_argnames=['is_training'])
    def loss_fn(params, state, batch, is_training=False):
        x_train, y_train = batch
        y_onehot = jax.nn.one_hot(y_train, num_classes=10)
        logits, state = apply_fn(params, state, x_train, is_training=is_training)
        softmax_xent = -jnp.sum(y_onehot * jax.nn.log_softmax(logits))
        softmax_xent /= len(x_train)
        return softmax_xent, state

    @jax.jit
    def acc_fn(params, state, batch):
        x_train, y_train = batch
        preds, _ = apply_fn(params, state, x_train, is_training=False)
        acc = jnp.mean(jnp.argmax(preds, axis=-1) == y_train)
        return acc
    
    return loss_fn, acc_fn

def get_train_step_fn(optimizer, loss_fn):
    
    @jax.jit
    def train_step_fn(params, state, batch, opt_state):
        (loss_value, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True, argnums=0)(
        params, state, batch, is_training=True)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return loss_value, new_state, new_params, opt_state
    
    return train_step_fn


