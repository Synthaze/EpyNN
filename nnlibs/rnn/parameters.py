# EpyNN/nnlibs/rnn/parameters.py
# Related third party imports
import numpy as np


def rnn_compute_shapes(layer, A):
    """Compute forward shapes and dimensions for cells and layer.
    """
    X = A    # Input of current layer of shape (m, s, v)

    layer.d['m'] = X.shape[0]    # Number of samples (m)
    layer.d['s'] = X.shape[1]    # Length of sequence (s)
    layer.d['v'] = X.shape[2]    # Vocabulary size (v)

    # Shapes for parameters to compute hidden cell state
    hv = layer.fs['U'] = (layer.d['h'], layer.d['v'])
    hh = layer.fs['W'] = (layer.d['h'], layer.d['h'])
    h1 = layer.fs['b'] = (layer.d['h'], 1)

    # Shapes to initialize caches
    msh = (layer.d['m'], layer.d['s'], layer.d['h'])

    layer.fs['h'] = layer.fs['A'] = msh

    return None


def rnn_initialize_parameters(layer):
    """Initialize parameters for layer.
    """
    # Parameters for cell output
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

    # Iterate through reversed sequence
    for s in reversed(range(layer.d['s'])):

        #
        dh = layer.bc['dh'][:, s]     # Current cell state error
        hp = layer.fc['h'][:, s - 1]  # Previous cell state
        X = layer.fc['X'][:, s]       # Current cell input
        # Gradients
        layer.g['dU'] += np.dot(X.T, dh)
        layer.g['dW'] += np.dot(hp.T, dh)
        layer.g['db'] += np.sum(dh, axis=0)

    return None


def rnn_update_parameters(layer):
    """Update parameters for layer.
    """
    for gradient in layer.g.keys():
        parameter = gradient[1:]
        #
        layer.p[parameter] -= layer.lrate[layer.e] * layer.g[gradient]

    return None
