# EpyNN/nnlibs/dense/parameters.py
# Related third party imports
import numpy as np


def dense_compute_shapes(layer, A):
    """Compute forward shapes and dimensions for layer.
    """
    X = A    # Input of current layer of shape (n, m)

    layer.fs['X'] = X.shape

#    layer.d['p'] = layer.fs['X'][0]
#    layer.d['m'] = layer.fs['X'][1]
    layer.d['m'] = layer.fs['X'][0]
    layer.d['p'] = layer.fs['X'][1]

#    nm = layer.fs['W'] = (layer.d['n'], layer.d['p'])
    nm = layer.fs['W'] = (layer.d['n'], layer.d['p'])

    n1 = layer.fs['b'] = (layer.d['n'], 1)

    return None


def dense_initialize_parameters(layer):
    """Initialize parameters for layer.
    """
    layer.p['W'] = layer.initialization(layer.fs['W'], rng=layer.np_rng)
    layer.p['b'] = np.zeros(layer.fs['b'])

    return None


def dense_compute_gradients(layer):
    """Compute gradients with respect to weight and bias for layer.
    """
    # X - Input of forward propagation
    X = layer.fc['X']
    # dZ - Gradient of the cost with respect to the linear output of forward propagation (Z)
    dZ = layer.bc['dZ']

    # dW - Gradient of the cost with respect to weight (W)
    dW = layer.g['dW'] = np.dot(X.T, dZ)
    # db - Gradient of the cost with respect to biais (b)
    db = layer.g['db'] = np.sum(dZ, axis=0)

    return None


def dense_update_parameters(layer):
    """Update parameters for layer.
    """
    for gradient in layer.g.keys():
        parameter = gradient[1:]
        #
        layer.p[parameter] -= layer.lrate[layer.e] * layer.g[gradient]

    return None
