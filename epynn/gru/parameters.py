# EpyNN/epynn/gru/parameters.py
# Related third party imports
import numpy as np


def gru_compute_shapes(layer, A):
    """Compute forward shapes and dimensions from input for layer.
    """
    X = A    # Input of current layer

    layer.fs['X'] = X.shape    # (m, s, e)

    layer.d['m'] = layer.fs['X'][0]    # Number of samples (m)
    layer.d['s'] = layer.fs['X'][1]    # Steps in sequence (s)
    layer.d['e'] = layer.fs['X'][2]    # Elements per step (e)

    # Parameter Shapes             Unit cells (u)
    eu = (layer.d['e'], layer.d['u'])    # (e, u)
    uu = (layer.d['u'], layer.d['u'])    # (u, u)
    u1 = (1, layer.d['u'])               # (1, u)
    # Update gate    Reset gate       Hidden hat
    layer.fs['Uz'] = layer.fs['Ur'] = layer.fs['Uhh'] = eu
    layer.fs['Vz'] = layer.fs['Vr'] = layer.fs['Vhh'] = uu
    layer.fs['bz'] = layer.fs['br'] = layer.fs['bhh'] = u1

    # Shape of hidden state (h) with respect to steps (s)
    layer.fs['h'] = (layer.d['m'], layer.d['s'], layer.d['u'])

    return None


def gru_initialize_parameters(layer):
    """Initialize trainable parameters from shapes for layer.
    """
    # For linear activation of update gate (z_)
    layer.p['Uz'] = layer.initialization(layer.fs['Uz'], rng=layer.np_rng)
    layer.p['Vz'] = layer.initialization(layer.fs['Vz'], rng=layer.np_rng)
    layer.p['bz'] = np.zeros(layer.fs['bz']) # dot(X, U) + dot(hp, V) + b

    # For linear activation of reset gate (r_)
    layer.p['Ur'] = layer.initialization(layer.fs['Ur'], rng=layer.np_rng)
    layer.p['Vr'] = layer.initialization(layer.fs['Vr'], rng=layer.np_rng)
    layer.p['br'] = np.zeros(layer.fs['br']) # dot(X, U) + dot(hp, V) + b

    # For linear activation of hidden hat (hh_)
    layer.p['Uhh'] = layer.initialization(layer.fs['Uhh'], rng=layer.np_rng)
    layer.p['Vhh'] = layer.initialization(layer.fs['Vhh'], rng=layer.np_rng)
    layer.p['bhh'] = np.zeros(layer.fs['bhh']) # dot(X, U) + dot(r * hp, V) + b

    return None


def gru_compute_gradients(layer):
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
        dhh_ = layer.bc['dhh_'][:, s]            # Gradient w.r.t hidden hat hh_
        layer.g['dUhh'] += np.dot(X.T, dhh_)     # (1.1) dL/dUhh
        layer.g['dVhh'] += np.dot((layer.fc['r'][:, s] * hp).T, dhh_)
        layer.g['dbhh'] += np.sum(dhh_, axis=0)  # (1.3) dL/dbhh

        # (2) Gradients of the loss with respect to U, V, b
        dz_ = layer.bc['dz_'][:, s]             # Gradient w.r.t update gate z_
        layer.g['dUz'] += np.dot(X.T, dz_)      # (2.1) dL/dUz
        layer.g['dVz'] += np.dot(hp.T, dz_)     # (2.2) dL/dVz
        layer.g['dbz'] += np.sum(dz_, axis=0)   # (2.3) dL/dbz

        # (3) Gradients of the loss with respect to U, V, b
        dr_ = layer.bc['dr_'][:, s]             # Gradient w.r.t reset gate r_
        layer.g['dUr'] += np.dot(X.T, dr_)      # (3.1) dL/dUr
        layer.g['dVr'] += np.dot(hp.T, dr_)     # (3.2) dL/dVr
        layer.g['dbr'] += np.sum(dr_, axis=0)   # (3.3) dL/dbr

    return None


def gru_update_parameters(layer):
    """Update parameters from gradients for layer.
    """
    for gradient in layer.g.keys():
        parameter = gradient[1:]
        # Update is driven by learning rate and gradients
        layer.p[parameter] -= layer.lrate[layer.e] * layer.g[gradient]

    return None
