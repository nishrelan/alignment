import jax
import jax.numpy as jnp
import optax
import sys
from functools import partial
import logging

log = logging.getLogger(__name__)

# Full gradient descent (only one batch)
def train(params, opt_state, train_step_fn, train_loader, test_loader, acc_fn, num_epochs, metrics):
    results = {m.name: [] for m in metrics}
    results['train_acc'] = []
    results['test_acc'] = []
    results['train_loss'] = []
    results['test_loss'] = []
    for epoch in range(num_epochs + 1):
        
        batch = next(iter(train_loader))
        test_batch = next(iter(test_loader))

        # Record metrics first and then update params at end of iteration
        for metric in metrics:
            if metric.interval and epoch % metric.interval == 0:
                results[metric.name].append(
                    (metric.fun(params=params, batch=batch), epoch)
                )

        epoch_acc = acc_fn(params, batch)
        test_acc = acc_fn(params, test_batch)
        epoch_test_loss = train_step_fn(params, opt_state, test_batch, just_loss=True)
        epoch_train_loss, params, opt_state = train_step_fn(params, opt_state, batch, just_loss=False)

        results['train_acc'].append((epoch_acc, epoch))
        results['test_acc'].append((test_acc, epoch))
        results['train_loss'].append((epoch_train_loss, epoch))
        results['test_loss'].append((epoch_test_loss, epoch))
        
        if epoch != 0:
            log.info("Epoch: {} Train Loss: {} Test Loss: {} Train Acc: {} Test Acc: {}".format(
                epoch, epoch_train_loss, epoch_test_loss, epoch_acc, test_acc))

        
            
    
    return results, params, opt_state



def get_mse_loss(model, init_params, alpha):

    def mse_loss(params, batch):
        xb, yb = batch
        preds = alpha*(model(params, xb) - model(init_params, xb))
        preds = jnp.ravel(preds)
        loss = jnp.mean((yb - preds)**2)
        return loss
    
    return mse_loss


def get_hinge_loss(model, init_params, alpha):
    
    @jax.jit
    def hinge_loss(params, batch):
        xb, yb = batch
        preds = alpha*(model(params, xb) - model(init_params, xb))
        preds = jnp.ravel(preds)
        preds = 1 - yb*preds
        return jnp.mean(jnp.where(preds < 0, 0, preds)) / alpha
    
    return hinge_loss

def get_update_fun(optimizer, loss_fn):

    @partial(jax.jit, static_argnames=['just_loss'])
    def train_step_fn(params, opt_state, batch, just_loss=False):
        if just_loss:
            return loss_fn(params, batch)
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, opt_state

    return train_step_fn


# TODO: fix this
def make_acc_fn(model, init_params, alpha):

    @jax.jit
    def acc(params, batch):
        xb, yb = batch
        raw_preds = alpha*(model(params, xb) - model(init_params, xb))
        raw_preds = jnp.ravel(raw_preds)
        preds = jnp.where(raw_preds > 0, 1, -1)
        return jnp.mean(preds == yb)
    
    return acc


# TODO: make functional
def predict(alpha, model, params, init_params, xb):
    raw_preds = alpha*(model(params, xb) - model(init_params, xb))
    raw_preds = jnp.ravel(raw_preds)
    preds = jnp.where(raw_preds > 0, 1, -1)
    return preds, raw_preds
