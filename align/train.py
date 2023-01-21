import jax
import jax.numpy as jnp
import optax
import sys


# Full gradient descent (only one batch)
def train(params, opt_state, train_step_fn, train_loader, test_loader, acc_fn, num_epochs, metrics):
    results = {m.name: [] for m in metrics}
    for epoch in range(1, num_epochs + 1):
        
        batch = next(iter(train_loader))
        epoch_loss, params, opt_state = train_step_fn(params, opt_state, batch)
        epoch_acc = acc_fn(params, batch)

        print("Epoch: {} Loss: {} Acc: {}".format(epoch, epoch_loss, epoch_acc))

        for metric in metrics:
            if metric.interval and epoch % metric.interval == 0:
                results[metric.name].append(
                    (metric.fun(params=params, batch=batch), epoch)
                )
            
    
    return results, params, opt_state



def get_hinge_loss(model, init_params, alpha):
    
    @jax.jit
    def hinge_loss(params, batch):
        xb, yb = batch
        preds = alpha*(model.apply(params, xb) - model.apply(init_params, xb))
        preds = jnp.ravel(preds)
        preds = 1 - yb*preds
        return jnp.mean(jnp.where(preds < 0, 0, preds)) / alpha
 
    
    return hinge_loss

def get_update_fun(optimizer, loss_fn):

    @jax.jit
    def train_step_fn(params, opt_state, batch):
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
        raw_preds = alpha*(model.apply(params, xb) - model.apply(init_params, xb))
        raw_preds = jnp.ravel(raw_preds)
        preds = jnp.where(raw_preds > 0, 1, -1)
        return jnp.mean(preds == yb)
    
    return acc


# TODO: make functional
def predict(alpha, model, params, init_params, xb):
    raw_preds = alpha*(model.apply(params, xb) - model.apply(init_params, xb))
    raw_preds = jnp.ravel(raw_preds)
    preds = jnp.where(raw_preds > 0, 1, -1)
    return preds, raw_preds