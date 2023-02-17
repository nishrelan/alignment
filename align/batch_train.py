import jax
import jax.numpy as jnp
import optax
import sys
from functools import partial
import logging

log = logging.getLogger(__name__)



def train(params, opt_state, train_step_fn, train_loader, test_loader, loss_fn, acc_fn, num_iters):
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


        train_acc = acc_fn(params, train_batch)
        test_acc = acc_fn(params, test_batch)
        test_loss = loss_fn(params, test_batch)
        train_loss, params, opt_state = train_step_fn(params, opt_state, train_batch)

        print(
            "Epoch: {} Train Loss: {} Test Loss: {} Train Acc: {} Test Acc: {}".format(
                i, train_loss, test_loss, train_acc, test_acc
            )
        )
        results['train_acc'].append(train_acc)
        results['test_acc'].append(test_acc)
        results['test_loss'].append(test_loss)
        results['train_loss'].append(train_loss)
    
    return results, params, opt_state


            


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


