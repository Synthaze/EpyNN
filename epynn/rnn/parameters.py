# EpyNN/epynn/rnn/parameters.py
# Related third party imports
import numpy as np


def rnn_compute_shapes(layer, A):
    """Compute forward shapes and dimensions from input for layer.
    """
    X = A    # Input of current layer

    layer.fs['X'] = X.shape    # (m, s, e)

    layer.d['m'] = layer.fs['X'][0]    # Number of samples (m)
    layer.d['s'] = layer.fs['X'][1]    # Steps in sequence (s)
    layer.d['e'] = layer.fs['X'][2]    # Elements per step (e)

    # Shapes for trainable parameters         Unit cells (u)
    layer.fs['U'] = (layer.d['e'], layer.d['u'])    # (e, u)
    layer.fs['V'] = (layer.d['u'], layer.d['u'])    # (u, u)
    layer.fs['b'] = (1, layer.d['u'])               # (1, u)

    # Shape of hidden state (h) with respect to steps (s)
    layer.fs['h'] = (layer.d['m'], layer.d['s'], layer.d['u'])

    return None


def rnn_initialize_parameters(layer):
    """Initialize trainable parameters from shapes for layer.
    """
    # For linear activation of hidden state (h_)
    layer.p['U'] = layer.initialization(layer.fs['U'], rng=layer.np_rng)
    layer.p['V'] = layer.initialization(layer.fs['V'], rng=layer.np_rng)
    layer.p['b'] = np.zeros(layer.fs['b']) # dot(X, U) + dot(hp, V) + b

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

        dh_ = layer.bc['dh_'][:, s]  # Gradient w.r.t hidden state h_
        X = layer.fc['X'][:, s]      # Input for current step
        hp = layer.fc['hp'][:, s]    # Previous hidden state

        # (1) Gradients of the loss with respect to U, V, b
        layer.g['dU'] += np.dot(X.T, dh_)     # (1.1) dL/dU
        layer.g['dV'] += np.dot(hp.T, dh_)    # (1.2) dL/dV
        layer.g['db'] += np.sum(dh_, axis=0)  # (1.3) dL/db

    return None


def rnn_update_parameters(layer):
    """Update parameters from gradients for layer.
    """
    for gradient in layer.g.keys():
        parameter = gradient[1:]
        # Update is driven by learning rate and gradients
        layer.p[parameter] -= layer.lrate[layer.e] * layer.g[gradient]

    return None
