# EpyNN/epynn/dense/parameters.py
# Related third party imports
import numpy as np


def dense_compute_shapes(layer, A):
    """Compute forward shapes and dimensions from input for layer.
    """
    X = A    # Input of current layer

    layer.fs['X'] = X.shape    # (m, n)

    layer.d['m'] = layer.fs['X'][0]    # Number of samples  (m)
    layer.d['n'] = layer.fs['X'][1]    # Number of features (n)

    # Shapes for trainable parameters              Units (u)
    layer.fs['W'] = (layer.d['n'], layer.d['u'])    # (n, u)
    layer.fs['b'] = (1, layer.d['u'])               # (1, u)

    return None


def dense_initialize_parameters(layer):
    """Initialize trainable parameters from shapes for layer.
    """
    # For linear activation of inputs (Z)
    layer.p['W'] = layer.initialization(layer.fs['W'], rng=layer.np_rng)
    layer.p['b'] = np.zeros(layer.fs['b']) # Z = dot(X, W) + b

    return None


def dense_compute_gradients(layer):
    """Compute gradients with respect to weight and bias for layer.
    """
    X = layer.fc['X']      # Input of forward propagation
    dZ = layer.bc['dZ']    # Gradient of the loss with respect to Z

    # (1) Gradient of the loss with respect to W, b
    dW = layer.g['dW'] = np.dot(X.T, dZ)       # (1.1) dL/dW
    db = layer.g['db'] = np.sum(dZ, axis=0)    # (1.2) dL/db

    return None


def dense_update_parameters(layer):
    """Update parameters from gradients for layer.
    """
    for gradient in layer.g.keys():
        parameter = gradient[1:]
        # Update is driven by learning rate and gradients
        layer.p[parameter] -= layer.lrate[layer.e] * layer.g[gradient]

    return None
