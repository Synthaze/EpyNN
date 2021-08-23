# EpyNN/nnlibs/rnn/parameters.py
# Related third party imports
import numpy as np


def rnn_compute_shapes(layer, A):
    """Compute forward shapes and dimensions from input for layer.
    """
    X = A    # Input of current layer

    layer.fs['X'] = X.shape    # (m, s, v)

    layer.d['m'] = layer.fs['X'][0]    # Number of samples  (m)
    layer.d['s'] = layer.fs['X'][1]    # Length of sequence (s)
    layer.d['v'] = layer.fs['X'][2]    # Vocabulary size    (v)

    # Shapes for trainable parameters         Unit cells (u)
    layer.fs['U'] = (layer.d['v'], layer.d['u'])    # (v, u)
    layer.fs['W'] = (layer.d['u'], layer.d['u'])    # (u, u)
    layer.fs['b'] = (1, layer.d['u'])               # (1, u)

    # Shape of hidden cell state (h) with respect to steps (s)
    layer.fs['h'] = (layer.d['m'], layer.d['s'], layer.d['u'])

    return None


def rnn_initialize_parameters(layer):
    """Initialize trainable parameters from shapes for layer.
    """
    # For linear activation of hidden cell state (h)
    layer.p['U'] = layer.initialization(layer.fs['U'], rng=layer.np_rng)
    layer.p['W'] = layer.initialization(layer.fs['W'], rng=layer.np_rng)
    layer.p['b'] = np.zeros(layer.fs['b']) # dot(X, U) + dot(hp, W) + b

    return None


def rnn_compute_gradients(layer):
    """Compute gradients with respect to weight and bias for layer.
    """
    # Gradients initialization with respect to parameters
    for parameter in layer.p.keys():
        gradient = 'd' + parameter
        layer.g[gradient] = np.zeros_like(layer.p[parameter])

    # Reverse iteration over sequence steps
    for s in reversed(range(layer.d['s'])):

        dh = layer.bc['dh'][:, s]    # Gradient for current hidden state
        X = layer.fc['X'][:, s]      # Current cell input
        hp = layer.fc['hp'][:, s]    # Previous hidden state

        # (1) Gradients of the loss with respect to U, W, b
        layer.g['dU'] += np.dot(X.T, dh)     # (1.1)
        layer.g['dW'] += np.dot(hp.T, dh)    # (1.2)
        layer.g['db'] += np.sum(dh, axis=0)  # (1.3)

    return None


def rnn_update_parameters(layer):
    """Update parameters from gradients for layer.
    """
    for gradient in layer.g.keys():
        parameter = gradient[1:]
        # Update is driven by learning rate and gradients
        layer.p[parameter] -= layer.lrate[layer.e] * layer.g[gradient]

    return None
