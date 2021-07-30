# EpyNN/nnlibs/rnn/parameters.py
# Related third party imports
import numpy as np


def rnn_compute_shapes(layer, A):
    """Compute forward shapes and dimensions for RNN cell and layer.
    """
    X = A    # Input of current layer of shape (s, v, m)

    layer.d['s'] = X.shape[0]    # Length of sequence (s)
    layer.d['v'] = X.shape[1]    # Vocabulary size (v)
    layer.d['m'] = X.shape[2]    # Number of samples (m)
    # Max length (l) between cells and sequence
    layer.d['l'] = max(layer.d['h'], layer.d['s'])
    # Output length (o)
    layer.d['o'] = 2 if layer.binary else layer.d['v']

    # Shapes for parameters to compute hidden cell state to next cell
    hv = layer.fs['Wx'] = (layer.d['h'], layer.d['v'])
    hh = layer.fs['Wh'] = (layer.d['h'], layer.d['h'])
    h1 = layer.fs['bh'] = (layer.d['h'], 1)
    # Shapes for parameters to compute cell output to next layer
    oh = layer.fs['W'] = (layer.d['o'], layer.d['h'])
    o1 = layer.fs['b'] = (layer.d['o'], 1)

    # Shapes to initialize caches
    lvm = layer.fs['X'] = (layer.d['l'], layer.d['v'], layer.d['m'])
    hhm = layer.fs['h'] = (layer.d['h'], layer.d['h'], layer.d['m'])
    ohm = layer.fs['A'] = (layer.d['h'], layer.d['o'], layer.d['m'])

    return None


def rnn_initialize_parameters(layer):
    """Initialize parameters for RNN layer.
    """
    # Parameters to compute cell state to next cell
    layer.p['Wx'] = layer.initialization(layer.fs['Wx'], rng=layer.np_rng)
    layer.p['Wh'] = layer.initialization(layer.fs['Wh'], rng=layer.np_rng)
    layer.p['bh'] = np.zeros(layer.fs['bh'])
    # Parameters to compute cell output to next layer
    layer.p['W'] = layer.initialization(layer.fs['W'], rng=layer.np_rng)
    layer.p['b'] = np.zeros(layer.fs['b'])

    return None


def rnn_compute_gradients(layer):
    """Update gradients with respect to weight and bias from RNN cells in layer.
    """
    # Gradients initialization with respect to parameters
    for parameter in layer.p.keys():
        gradient = 'd' + parameter
        layer.g[gradient] = np.zeros_like(layer.p[parameter])

    # Iterate through reversed sequence
    for s in reversed(range(layer.d['h'])):

        h = layer.fc['h'][s]    # Current cell state
        dX = layer.bc['dX'] if layer.binary else layer.bc['dX'][s]
        # Gradients with respect to layer output
        layer.g['dW'] += np.dot(dX, h.T)
        layer.g['db'] += np.sum(dX, axis=1, keepdims=True)

        X = layer.fc['X'][s]       # Current cell input
        dh = layer.bc['dh'][s]     # Current cell state error
        hp = layer.fc['h'][s - 1]  # Previous cell state
        # Gradients with respect to cell output
        layer.g['dWx'] += np.dot(dh, X.T)
        layer.g['dWh'] += np.dot(dh, hp.T)
        layer.g['dbh'] += np.sum(dh, axis=1, keepdims=True)

    return None


def rnn_update_parameters(layer):
    """Update parameters for RNN layer.
    """
    for gradient in layer.g.keys():
        parameter = gradient[1:]
        layer.p[parameter] -= layer.lrate[layer.e] * layer.g[gradient]

    return None
