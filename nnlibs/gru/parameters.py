# EpyNN/nnlibs/gru/parameters.py
# Related third party imports
import numpy as np


def gru_compute_shapes(layer, A):
    """Compute forward shapes and dimensions for cells and layer.
    """
    X = A    # Input of current layer

    layer.fs['X'] = X.shape    # (m, s, v)

    layer.d['m'] = layer.fs['X'][0]    # Number of samples (m)
    layer.d['s'] = layer.fs['X'][1]    # Length of sequence (s)
    layer.d['v'] = layer.fs['X'][2]    # Vocabulary size (v)

    # Parameters shape
    vh = (layer.d['v'], layer.d['h'])
    hh = (layer.d['h'], layer.d['h'])
    h1 = (layer.d['h'],)

    # U applies to X - Gate/Activation-specific
    layer.fs['Uz'] = layer.fs['Ur'] = layer.fs['Uh'] = vh
    # Wz and Wr applies to hp - Wh applies to r * hp
    layer.fs['Wz'] = layer.fs['Wr'] = layer.fs['Wh'] = hh
    # b is added to the linear activation product - Gate/Activation-specific
    layer.fs['bz'] = layer.fs['br'] = layer.fs['bh'] = h1

    # Shape of cache for hidden cell states (h)
    msh = layer.fs['h'] = (layer.d['m'], layer.d['s'], layer.d['h'])

    return None


def gru_initialize_parameters(layer):
    """Initialize parameters for layer.
    """
    # U, W, b - Update gate
    layer.p['Uz'] = layer.initialization(layer.fs['Uz'], rng=layer.np_rng)
    layer.p['Wz'] = layer.initialization(layer.fs['Wz'], rng=layer.np_rng)
    layer.p['bz'] = np.zeros(layer.fs['bz'])

    # U, W, b - Reset gate
    layer.p['Ur'] = layer.initialization(layer.fs['Ur'], rng=layer.np_rng)
    layer.p['Wr'] = layer.initialization(layer.fs['Wr'], rng=layer.np_rng)
    layer.p['br'] = np.zeros(layer.fs['br'])

    # U, W, b - Hidden (hh) cell state activation
    layer.p['Uh'] = layer.initialization(layer.fs['Uh'], rng=layer.np_rng)
    layer.p['Wh'] = layer.initialization(layer.fs['Wh'], rng=layer.np_rng)
    layer.p['bh'] = np.zeros(layer.fs['bh'])

    return None


def gru_compute_gradients(layer):
    """Compute gradients with respect to weight and bias for cells and layer.
    """
    # Gradients initialization with respect to parameters
    for parameter in layer.p.keys():
        gradient = 'd' + parameter
        layer.g[gradient] = np.zeros_like(layer.p[parameter])

    # Iterate over reversed sequence steps
    for s in reversed(range(layer.d['s'])):

        # Retrieve from layer cache
        X = layer.fc['X'][:, s]       # Current cell input
        hp = layer.fc['hp'][:, s]     # Previous cell state

        # (1) Gradients with respect to U, W, b
        dhh = layer.bc['dhh'][:, s]  # For hidden hat (hh) activation
        layer.g['dUh'] += np.dot(X.T, dhh)     # (1.1)
        layer.g['dWh'] += np.dot((layer.fc['r'][:, s] * hp).T, dhh)
        layer.g['dbh'] += np.sum(dhh, axis=0)  # (1.3)

        # (2) Gradients with respect to U, W, b
        dz = layer.bc['dz'][:, s]  # For update gate
        layer.g['dUz'] += np.dot(X.T, dz)     # (2.1)
        layer.g['dWz'] += np.dot(hp.T, dz)    # (2.2)
        layer.g['dbz'] += np.sum(dz, axis=0)  # (2.3)

        # (3) Gradients with respect to U, W, b
        dr = layer.bc['dr'][:, s]  # For reset gate
        layer.g['dUr'] += np.dot(X.T, dr)     # (3.1)
        layer.g['dWr'] += np.dot(hp.T, dr)    # (3.2)
        layer.g['dbr'] += np.sum(dr, axis=0)  # (3.3)

    return None


def gru_update_parameters(layer):
    """Update parameters for layer.
    """
    for gradient in layer.g.keys():
        parameter = gradient[1:]
        # Update is driven by learning rate and gradients
        layer.p[parameter] -= layer.lrate[layer.e] * layer.g[gradient]

    return None
