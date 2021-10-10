# EpyNN/epynn/lstm/parameters.py
# Related third party imports
import numpy as np


def lstm_compute_shapes(layer, A):
    """Compute forward shapes and dimensions from input for layer.
    """
    X = A    # Input of current layer

    layer.fs['X'] = X.shape    # (m, s, e)

    layer.d['m'] = layer.fs['X'][0]    # Number of samples (m)
    layer.d['s'] = layer.fs['X'][1]    # Steps in sequence (s)
    layer.d['e'] = layer.fs['X'][2]    # Elements per step (e)

    # Parameter Shapes             Unit cells (u)
    eu = (layer.d['e'], layer.d['u'])    # (v, u)
    uu = (layer.d['u'], layer.d['u'])    # (u, u)
    u1 = (1, layer.d['u'])               # (1, u)
    # Forget gate    Input gate       Candidate        Output gate
    layer.fs['Uf'] = layer.fs['Ui'] = layer.fs['Ug'] = layer.fs['Uo'] = eu
    layer.fs['Vf'] = layer.fs['Vi'] = layer.fs['Vg'] = layer.fs['Vo'] = uu
    layer.fs['bf'] = layer.fs['bi'] = layer.fs['bg'] = layer.fs['bo'] = u1

    # Shape of hidden (h) and memory (C) state with respect to steps (s)
    layer.fs['h'] = layer.fs['C'] = (layer.d['m'], layer.d['s'], layer.d['u'])

    return None


def lstm_initialize_parameters(layer):
    """Initialize trainable parameters from shapes for layer.
    """
    # For linear activation of forget gate (f_)
    layer.p['Uf'] = layer.initialization(layer.fs['Uf'], rng=layer.np_rng)
    layer.p['Vf'] = layer.initialization(layer.fs['Vf'], rng=layer.np_rng)
    layer.p['bf'] = np.zeros(layer.fs['bf']) # dot(X, U) + dot(hp, V) + b

    # For linear activation of input gate (i_)
    layer.p['Ui'] = layer.initialization(layer.fs['Ui'], rng=layer.np_rng)
    layer.p['Vi'] = layer.initialization(layer.fs['Vi'], rng=layer.np_rng)
    layer.p['bi'] = np.zeros(layer.fs['bi']) # dot(X, U) + dot(hp, V) + b

    # For linear activation of candidate (g_)
    layer.p['Ug'] = layer.initialization(layer.fs['Ug'], rng=layer.np_rng)
    layer.p['Vg'] = layer.initialization(layer.fs['Vg'], rng=layer.np_rng)
    layer.p['bg'] = np.zeros(layer.fs['bg']) # dot(X, U) + dot(hp, V) + b

    # For linear activation of output gate (o_)
    layer.p['Uo'] = layer.initialization(layer.fs['Uo'], rng=layer.np_rng)
    layer.p['Vo'] = layer.initialization(layer.fs['Vo'], rng=layer.np_rng)
    layer.p['bo'] = np.zeros(layer.fs['bo']) # dot(X, U) + dot(hp, V) + b

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

        X = layer.fc['X'][:, s]      # Input for current step
        hp = layer.fc['hp'][:, s]    # Previous hidden state

        # (1) Gradients of the loss with respect to U, V, b
        do_ = layer.bc['do_'][:, s]            # Gradient w.r.t output gate o_
        layer.g['dUo'] += np.dot(X.T, do_)     # (1.1) dL/dUo
        layer.g['dVo'] += np.dot(hp.T, do_)    # (1.2) dL/dVo
        layer.g['dbo'] += np.sum(do_, axis=0)  # (1.3) dL/dbo

        # (2) Gradients of the loss with respect to U, V, b
        dg_ = layer.bc['dg_'][:, s]            # Gradient w.r.t candidate g_
        layer.g['dUg'] += np.dot(X.T, dg_)     # (2.1) dL/dUg
        layer.g['dVg'] += np.dot(hp.T, dg_)    # (2.2) dL/dVg
        layer.g['dbg'] += np.sum(dg_, axis=0)  # (2.3) dL/dbg

        # (3) Gradients of the loss with respect to U, V, b
        di_ = layer.bc['di_'][:, s]            # Gradient w.r.t input gate i_
        layer.g['dUi'] += np.dot(X.T, di_)     # (3.1) dL/dUi
        layer.g['dVi'] += np.dot(hp.T, di_)    # (3.2) dL/dVi
        layer.g['dbi'] += np.sum(di_, axis=0)  # (3.3) dL/dbi

        # (4) Gradients of the loss with respect to U, V, b
        df_ = layer.bc['df_'][:, s]            # Gradient w.r.t forget gate f_
        layer.g['dUf'] += np.dot(X.T, df_)     # (4.1) dL/dUf
        layer.g['dVf'] += np.dot(hp.T, df_)    # (4.2) dL/dVf
        layer.g['dbf'] += np.sum(df_, axis=0)  # (4.3) dL/dbf

    return None


def lstm_update_parameters(layer):
    """Update parameters from gradients for layer.
    """
    for gradient in layer.g.keys():
        parameter = gradient[1:]
        # Update is driven by learning rate and gradients
        layer.p[parameter] -= layer.lrate[layer.e] * layer.g[gradient]

    return None
