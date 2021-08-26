# EpyNN/nnlibs/convolution/parameters.py
# Standard library imports
import math

# Related third party imports
import numpy as np


def convolution_compute_shapes(layer, A):
    """Compute forward shapes and dimensions for layer.
    """
    X = A    # Input of current layer

    layer.fs['X'] = X.shape         # (m, h, w, d)

    layer.d['m'] = layer.fs['X'][0]     # Number of samples  (m)
    layer.d['h'] = layer.fs['X'][1]     # Height of features (h)
    layer.d['w'] = layer.fs['X'][2]     # Width of features  (w)
    layer.d['d'] = layer.fs['X'][3]     # Depth of features  (d)

    # Shapes for trainable parameters
    # filter_height (fh), filter_width (fw), features_depth (d), unit_filters (u)
    layer.fs['W'] = (layer.d['fh'], layer.d['fw'], layer.d['d'], layer.d['u'])
    layer.fs['b'] = (layer.d['u'], )

    return None


def convolution_initialize_parameters(layer):
    """Initialize parameters for layer.
    """
    # For linear activation of inputs (Z)
    layer.p['W'] = layer.initialization(layer.fs['W'], rng=layer.np_rng)
    layer.p['b'] = np.zeros(layer.fs['b']) # Z = X * W + b

    return None


def convolution_compute_gradients(layer):
    """Compute gradients with respect to weight and bias for layer.
    """
    # Gradients initialization with respect to parameters
    for parameter in layer.p.keys():
        gradient = 'd' + parameter
        layer.g[gradient] = np.zeros_like(layer.p[parameter])

    Xb = layer.fc['Xb']     # Input blocks of forward propagation
    dZ = layer.bc['dZ']     # Gradient of the loss with respect to Z

    # Expand dZ dimensions with respect to Xb
    dZ = np.expand_dims(dZ, axis=3)
    dZ = np.expand_dims(dZ, axis=3)
    dZ = np.expand_dims(dZ, axis=3)

    # (1) Gradients of the loss with respect to W
    dW = dZ * Xb               # (1.1)
    dW = np.sum(dW, axis=2)    # (1.2.1)
    dW = np.sum(dW, axis=1)    # (1.2.2)
    dW = np.sum(dW, axis=0)    # (1.2.3)

    layer.g['dW'] = dW

    # (2) Gradients of the loss with respect to b
    db = dW                    # (2.1)
    db = np.sum(db, axis=2)    # (2.2.1)
    db = np.sum(db, axis=1)    # (2.2.2)
    db = np.sum(db, axis=0)    # (2.2.3)

    layer.g['db'] = db if layer.use_bias else 0.

    return None


def convolution_update_parameters(layer):
    """Update parameters for layer.
    """
    for gradient in layer.g.keys():
        parameter = gradient[1:]
        # Update is driven by learning rate and gradients
        layer.p[parameter] -= layer.lrate[layer.e] * layer.g[gradient]

    return None
