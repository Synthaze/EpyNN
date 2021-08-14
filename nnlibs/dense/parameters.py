# EpyNN/nnlibs/dense/parameters.py
# Related third party imports
import numpy as np


def dense_compute_shapes(layer, A):
    """Compute forward shapes and dimensions for layer.
    """
    X = A    # Input of current layer

    layer.fs['X'] = X.shape    # (m, p)

    layer.d['m'] = layer.fs['X'][0]    # Number of samples (m)
    layer.d['p'] = layer.fs['X'][1]    # Number of nodes previous layer (p)

    # Apply to X - W shape is (nodes_previous_layer, nodes_current)
    nm = layer.fs['W'] = (layer.d['p'], layer.d['n'])
    n1 = layer.fs['b'] = (1, layer.d['n'])

    return None


def dense_initialize_parameters(layer):
    """Initialize parameters for layer.
    """
    # W, b - Linear activation X -> Z
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

    # (1)
    dW = layer.g['dW'] = np.dot(X.T, dZ)     # (1.1)
    db = layer.g['db'] = np.sum(dZ, axis=0)  # (1.2)

    return None


def dense_update_parameters(layer):
    """Update parameters for layer.
    """
    for gradient in layer.g.keys():
        parameter = gradient[1:]
        # Update is driven by learning rate and gradients
        layer.p[parameter] -= layer.lrate[layer.e] * layer.g[gradient]

    return None
