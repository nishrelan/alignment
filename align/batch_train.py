import jax
import jax.numpy as jnp
import optax
import sys
from functools import partial
import logging

log = logging.getLogger(__name__)



def train(params, optimizer, opt_state, train_loader, test_loader, loss_fn, acc_fn, num_iters, print_iter=10):
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


       
        train_loss, params, opt_state = train_step_fn(loss_fn, optimizer, params, train_batch, opt_state)

        if i % print_iter == 0:
            test_acc = loader_stats(test_loader, acc_fn, params)
            train_acc = loader_stats(train_loader, acc_fn, params)
            train_loss = loader_stats(train_loader, loss_fn, params)
            test_loss = loader_stats(test_loader, loss_fn, params)
            log.info(
                "Epoch: {} Train Loss: {} Test Loss: {} Train Acc: {} Test Acc: {}".format(
                    i, train_loss, test_loss, train_acc, test_acc
                )
            )
        
    
    return results, params, opt_state


@partial(jax.jit, static_argnames=['loss_fn', 'optimizer'])
def train_step_fn(loss_fn, optimizer, params, batch, opt_state):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, opt_state

def get_bce_loss_acc(apply_fn):

    @jax.jit
    def loss_fn(params, batch):
        x, y = batch
        logits = apply_fn(params, x)
        logits = jnp.concatenate([logits, jnp.ones((logits.shape[0], 1))], axis=1)
        y_onehot = jax.nn.one_hot(y, num_classes=2)
        softmax_xent = -jnp.sum(y_onehot * jax.nn.log_softmax(logits))
        softmax_xent /= len(x)
        return softmax_xent

    @jax.jit
    def acc_fn(params, batch):
        x, y = batch
        preds = apply_fn(params, x)
        preds = jnp.concatenate([preds, jnp.ones((preds.shape[0], 1))], axis=1)
        acc = jnp.mean(jnp.argmax(preds, axis=-1) == y)
        return acc

    return loss_fn, acc_fn

def loader_stats(loader, fn, params):
    stats = 0.0
    num_total = 0
    for batch in loader:
        num = len(batch[0])
        avg = fn(params, batch)
        stats += avg*num
        num_total += num

    return stats / num_total


def get_mse_loss_acc(apply_fn):

    @jax.jit
    def loss_fn(params, batch):
        x, y = batch
        output = jnp.ravel(apply_fn(params, x))
        loss = jnp.sum((output - y)**2) / len(x)
        return loss

    @jax.jit
    def acc_fn(params, batch):
        x, y = batch
        output = jnp.ravel(apply_fn(params, x))
        preds = jnp.where(output >= 0.5, 1, 0)
        return jnp.mean(preds == y)
    
    return loss_fn, acc_fn


def get_xent_loss_acc(apply_fn):

    @jax.jit
    def loss_fn(params, batch):
        x_train, y_train = batch
        y_onehot = jax.nn.one_hot(y_train, num_classes=10)
        logits = apply_fn(params, x_train)
        softmax_xent = -jnp.sum(y_onehot * jax.nn.log_softmax(logits))
        softmax_xent /= len(x_train)
        return softmax_xent

    @jax.jit
    def acc_fn(params, batch):
        x_train, y_train = batch
        preds = apply_fn(params, x_train)
        acc = jnp.mean(jnp.argmax(preds, axis=-1) == y_train)
        return acc
    
    return loss_fn, acc_fn


