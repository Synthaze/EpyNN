# EpyNN/nnlibs/lstm/parameters.py
# Related third party imports
import numpy as np


def lstm_compute_shapes(layer, A):
    """Compute forward shapes and dimensions from input for layer.
    """
    X = A    # Input of current layer

    layer.fs['X'] = X.shape    # (m, s, v)

    layer.d['m'] = layer.fs['X'][0]    # Number of samples  (m)
    layer.d['s'] = layer.fs['X'][1]    # Length of sequence (s)
    layer.d['v'] = layer.fs['X'][2]    # Vocabulary size    (v)

    # Parameter Shapes             Unit cells (u)
    vu = (layer.d['v'], layer.d['u'])    # (v, u)
    uu = (layer.d['u'], layer.d['u'])    # (u, u)
    u1 = (1, layer.d['u'])               # (1, u)
    # Forget gate    Input gate       Candidate        Output gate
    layer.fs['Uf'] = layer.fs['Ui'] = layer.fs['Ug'] = layer.fs['Uo'] = vu
    layer.fs['Wf'] = layer.fs['Wi'] = layer.fs['Wg'] = layer.fs['Wo'] = uu
    layer.fs['bf'] = layer.fs['bi'] = layer.fs['bg'] = layer.fs['bo'] = u1

    # Shape of hidden (h) and memory (C) cell state with respect to steps (s)
    layer.fs['h'] = layer.fs['C'] = (layer.d['m'], layer.d['s'], layer.d['u'])

    return None


def lstm_initialize_parameters(layer):
    """Initialize trainable parameters from shapes for layer.
    """
    # For linear activation of forget gate (f)
    layer.p['Uf'] = layer.initialization(layer.fs['Uf'], rng=layer.np_rng)
    layer.p['Wf'] = layer.initialization(layer.fs['Wf'], rng=layer.np_rng)
    layer.p['bf'] = np.zeros(layer.fs['bf']) # dot(X, U) + dot(hp, W) + b

    # For linear activation of input gate (i)
    layer.p['Ui'] = layer.initialization(layer.fs['Ui'], rng=layer.np_rng)
    layer.p['Wi'] = layer.initialization(layer.fs['Wi'], rng=layer.np_rng)
    layer.p['bi'] = np.zeros(layer.fs['bi']) # dot(X, U) + dot(hp, W) + b

    # For linear activation of candidate (g)
    layer.p['Ug'] = layer.initialization(layer.fs['Ug'], rng=layer.np_rng)
    layer.p['Wg'] = layer.initialization(layer.fs['Wg'], rng=layer.np_rng)
    layer.p['bg'] = np.zeros(layer.fs['bg']) # dot(X, U) + dot(hp, W) + b

    # For linear activation of output gate (o)
    layer.p['Uo'] = layer.initialization(layer.fs['Uo'], rng=layer.np_rng)
    layer.p['Wo'] = layer.initialization(layer.fs['Wo'], rng=layer.np_rng)
    layer.p['bo'] = np.zeros(layer.fs['bo']) # dot(X, U) + dot(hp, W) + b

    return None


def lstm_compute_gradients(layer):
    """Compute gradients with respect to weight and bias for layer.
    """
    # Gradients initialization with respect to parameters
    for parameter in layer.p.keys():
        gradient = 'd' + parameter
        layer.g[gradient] = np.zeros_like(layer.p[parameter])

    # Reverse iteration over sequence steps
    for s in reversed(range(layer.d['s'])):

        X = layer.fc['X'][:, s]      # Current cell input
        hp = layer.fc['hp'][:, s]    # Previous hidden state

        # (1) Gradients of the loss with respect to U, W, b
        do = layer.bc['do'][:, s]             # Gradient output gate
        layer.g['dUo'] += np.dot(X.T, do)     # (1.1)
        layer.g['dWo'] += np.dot(hp.T, do)    # (1.2)
        layer.g['dbo'] += np.sum(do, axis=0)  # (1.3)

        # (2) Gradients of the loss with respect to U, W, b
        dg = layer.bc['dg'][:, s]             # Gradient candidate
        layer.g['dUg'] += np.dot(X.T, dg)     # (2.1)
        layer.g['dWg'] += np.dot(hp.T, dg)    # (2.2)
        layer.g['dbg'] += np.sum(dg, axis=0)  # (2.3)

        # (3) Gradients of the loss with respect to U, W, b
        di = layer.bc['di'][:, s]             # Gradient input gate
        layer.g['dUi'] += np.dot(X.T, di)     # (3.1)
        layer.g['dWi'] += np.dot(hp.T, di)    # (3.2)
        layer.g['dbi'] += np.sum(di, axis=0)  # (3.3)

        # (4) Gradients of the loss with respect to U, W, b
        df = layer.bc['df'][:, s]             # Gradient forget gate
        layer.g['dUf'] += np.dot(X.T, df)     # (4.1)
        layer.g['dWf'] += np.dot(hp.T, df)    # (4.2)
        layer.g['dbf'] += np.sum(df, axis=0)  # (4.3)

    return None


def lstm_update_parameters(layer):
    """Update parameters from gradients for layer.
    """
    for gradient in layer.g.keys():
        parameter = gradient[1:]
        # Update is driven by learning rate and gradients
        layer.p[parameter] -= layer.lrate[layer.e] * layer.g[gradient]

    return None
