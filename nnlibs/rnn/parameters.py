# EpyNN/nnlibs/rnn/parameters.py
# Related third party imports
import numpy as np


def rnn_compute_shapes(layer, A):
    """Compute forward shapes and dimensions for cells and layer.
    """
    X = A    # Input of current layer

    layer.fs['X'] = X.shape    # (m, s, v)

    layer.d['m'] = layer.fs['X'][0]    # Number of samples (m)
    layer.d['s'] = layer.fs['X'][1]    # Length of sequence (s)
    layer.d['v'] = layer.fs['X'][2]    # Vocabulary size (v)

    # Parameters shape
    vh = layer.fs['U'] = (layer.d['v'], layer.d['h'])  # U applies to X
    hh = layer.fs['W'] = (layer.d['h'], layer.d['h'])  # W applies to hp
    h1 = layer.fs['b'] = (layer.d['h'],)  # Added to the linear activation product

    # Shape of cache for hidden cell states
    msh = layer.fs['h'] = (layer.d['m'], layer.d['s'], layer.d['h'])

    return None


def rnn_initialize_parameters(layer):
    """Initialize parameters for layer.
    """
    # U, W, b - Hidden cell state activation
    layer.p['U'] = layer.initialization(layer.fs['U'], rng=layer.np_rng)
    layer.p['W'] = layer.initialization(layer.fs['W'], rng=layer.np_rng)
    layer.p['b'] = np.zeros(layer.fs['b'])

    return None


def rnn_compute_gradients(layer):
    """Compute gradients with respect to weight and bias for cells and layer.
    """
    # Gradients initialization with respect to parameters
    for parameter in layer.p.keys():
        gradient = 'd' + parameter
        layer.g[gradient] = np.zeros_like(layer.p[parameter])

    # Iterate over reversed sequence steps
    for s in reversed(range(layer.d['s'])):

        # Retrieve from layer cache
        dh = layer.bc['dh'][:, s]     # Current cell state error
        hp = layer.fc['h'][:, s - 1]  # Previous cell state
        X = layer.fc['X'][:, s]       # Current cell input

        # (1) Gradients with respect to U, W, b
        layer.g['dU'] += np.dot(X.T, dh)     # (1.1)
        layer.g['dW'] += np.dot(hp.T, dh)    # (1.2)
        layer.g['db'] += np.sum(dh, axis=0)  # (1.3)

    return None


def rnn_update_parameters(layer):
    """Update parameters for layer.
    """
    for gradient in layer.g.keys():
        parameter = gradient[1:]
        # Update is driven by learning rate and gradients
        layer.p[parameter] -= layer.lrate[layer.e] * layer.g[gradient]

    return None
