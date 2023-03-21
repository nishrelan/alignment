import haiku as hk
import jax
import jax.numpy as jnp
from functools import partial
from neural_tangents import stax


# Define network
class ConvLayer(hk.Module):
    def __init__(self, name, out_channels, kernel_shape, stride=1, data_format='NCHW', use_bn=True):
        if name is None:
            super().__init__()
        else:
            super().__init__(name)
        self.conv_layer = hk.Conv2D(out_channels, kernel_shape, 
                                        padding='SAME', stride=stride, data_format=data_format)
        self.use_bn = use_bn
        if self.use_bn:
            self.bn_layer = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.999, data_format='NCHW')

    def __call__(self, x, is_training):
        x = self.conv_layer(x)
        if self.use_bn:
            x = self.bn_layer(x, is_training)
        x = jax.nn.relu(x)
        return x

class CNN(hk.Module):
    def __init__(self, name, architecture, data_format='NCHW', use_bn=True):
        if name is None:
            super().__init__()
        else:
            super().__init__(name=name)
        
        channel_axis= hk.get_channel_index(data_format)
        self.layers = []
        for i, layer in enumerate(architecture):
            if isinstance(layer, int):
                self.layers.append(ConvLayer(None, layer, (3, 3), use_bn=use_bn))
            elif layer == 'M':
                self.layers.append(hk.MaxPool(window_shape=2, strides=2, padding='VALID', channel_axis=channel_axis))
        
        self.dense = hk.Linear(10)

    def __call__(self, x, is_training=True):
        for layer in self.layers:
            if isinstance(layer, ConvLayer):
                 x = layer(x, is_training=is_training)
            elif isinstance(layer, hk.MaxPool):
                x = layer(x)
        x = hk.Flatten()(x)
        x = self.dense(x)
        return x
        
def cnn_forward(x, name, architecture, is_training):
    model = CNN(name, architecture)
    return model(x, is_training=is_training)

def cnn_forward_no_bn(x, name, architecture):
    model = CNN(name, architecture, use_bn=False)
    return model(x)

def create_model(name, architecture, rng_key, sample_data, use_bn=True):
    if use_bn:
        model = hk.without_apply_rng(hk.transform_with_state(partial(cnn_forward, name=name, architecture=architecture)))
        init_params, init_state = model.init(rng_key, sample_data, is_training=True)
        return model, init_params, init_state
    else:
        model = hk.without_apply_rng(hk.transform(partial(cnn_forward_no_bn, name=name, architecture=architecture)))
        init_params = model.init(rng_key, sample_data)
        return model, init_params
    
def create_myrtle():
    W_std = 1.0
    b_std = 0.0
    
    