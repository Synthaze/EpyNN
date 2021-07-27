# EpyNN/nnlibs/rnn/parameters.py
# Related third party imports
import numpy as np


def rnn_compute_shapes(layer, A):
    """Compute forward shapes and dimensions for RNN cells and layer.
    """
    X = A    # Input of current layer

    layer.fs['X'] = X.shape    # (v, s, m)

    layer.d['v'] = layer.fs['X'][0]    # Vocabulary size
    layer.d['s'] = layer.fs['X'][1]    # Length of sequence
    layer.d['m'] = layer.fs['X'][2]    # Number of samples
    # Output length
    layer.d['o'] = 2 if layer.binary else layer.d['v']

    # Shapes for parameters to compute hidden cell state to next cell
    hv = layer.fs['Wx'] = (layer.d['h'], layer.d['v'])
    hh = layer.fs['Wh'] = (layer.d['h'], layer.d['h'])
    h1 = layer.fs['bh'] = (layer.d['h'], 1)
    # Shapes for parameters to compute cell output to next layer
    oh = layer.fs['W'] = (layer.d['o'], layer.d['h'])
    o1 = layer.fs['b'] = (layer.d['o'], 1)

    # Shapes to initialize forward cache
    svm = layer.fs['Xs'] = (layer.d['s'], layer.d['v'], layer.d['m'])
    shm = layer.fs['h'] = (layer.d['s'], layer.d['h'], layer.d['m'])
    som = layer.fs['A'] = (layer.d['s'], layer.d['o'], layer.d['m'])
    hm = layer.fs['hs'] = (layer.d['h'], layer.d['m'])

    return None


def rnn_initialize_parameters(layer):
    """Initialize parameters for RNN layer.
    """
    # Parameters to compute hidden cell state to next cell
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

    # Step through reversed sequence
    for s in reversed(range(layer.d['s'])):

        #
        h = layer.fc['h'][s]
        dXs = layer.bc['dXs'] if layer.binary else layer.bc['dXs'][s]
        # Gradients
        layer.g['dW'] += 1./ layer.d['m'] * np.dot(dXs, h.T)
        layer.g['db'] += 1./ layer.d['m'] * np.sum(dXs, axis=1, keepdims=True)

        #
        Xs = layer.fc['Xs'][s]
        hp = layer.fc['h'][s - 1]
        df = layer.bc['df'][s]
        # Gradients
        layer.g['dWx'] += 1./ layer.d['m'] * np.dot(df, Xs.T)
        layer.g['dWh'] += 1./ layer.d['m'] * np.dot(df, hp.T)
        layer.g['dbh'] += 1./ layer.d['m'] * np.sum(df, axis=1, keepdims=True)

    return None


def rnn_update_parameters(layer):
    """Update parameters for RNN layer.
    """
    for gradient in layer.g.keys():

        parameter = gradient[1:]
        
        layer.p[parameter] -= layer.lrate[layer.e] * layer.g[gradient]

    return None
