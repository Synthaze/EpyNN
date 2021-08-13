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

    vh = (layer.d['v'], layer.d['h'])
    hh = (layer.d['h'], layer.d['h'])
    h1 = (layer.d['h'],)

    layer.fs['Uz'] = layer.fs['Ur'] = layer.fs['Uh'] = vh
    layer.fs['Wz'] = layer.fs['Wr'] = layer.fs['Wh'] = hh
    layer.fs['bz'] = layer.fs['br'] = layer.fs['bh'] = h1

    # Shapes to initialize backward cache
    msh = layer.fs['h'] = (layer.d['m'], layer.d['s'], layer.d['h'])

    return None


def gru_initialize_parameters(layer):
    """Initialize parameters for layer.
    """
    #
    layer.p['Uz'] = layer.initialization(layer.fs['Uz'], rng=layer.np_rng)
    layer.p['Wz'] = layer.initialization(layer.fs['Wz'], rng=layer.np_rng)
    layer.p['bz'] = np.zeros(layer.fs['bz'])
    #
    layer.p['Ur'] = layer.initialization(layer.fs['Ur'], rng=layer.np_rng)
    layer.p['Wr'] = layer.initialization(layer.fs['Wr'], rng=layer.np_rng)
    layer.p['br'] = np.zeros(layer.fs['br'])
    #
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

    # Iterate through reversed sequence
    for s in reversed(range(layer.d['s'])):

        #
        X = layer.fc['X'][:, s]       # Current cell input
        hp = layer.fc['h'][:, s - 1]  # Previous cell state

        # (1)
        dhh = layer.bc['dhh'][:, s]
        layer.g['dUh'] += np.dot(X.T, dhh)     # (1.1)
        layer.g['dWh'] += np.dot((layer.fc['r'][:, s] * hp).T, dhh)
        layer.g['dbh'] += np.sum(dhh, axis=0)  # (1.3)

        # (2)
        dz = layer.bc['dz'][:, s]
        layer.g['dUz'] += np.dot(X.T, dz)     # (2.1)
        layer.g['dWz'] += np.dot(hp.T, dz)    # (2.2)
        layer.g['dbz'] += np.sum(dz, axis=0)  # (2.3)

        # (3)
        dr = layer.bc['dr'][:, s]
        layer.g['dUr'] += np.dot(X.T, dr)     # (3.1)
        layer.g['dWr'] += np.dot(hp.T, dr)    # (3.2)
        layer.g['dbr'] += np.sum(dr, axis=0)  # (3.3)

    return None


def gru_update_parameters(layer):
    """Update parameters for layer.
    """
    for gradient in layer.g.keys():
        parameter = gradient[1:]
        #
        layer.p[parameter] -= layer.lrate[layer.e] * layer.g[gradient]

    return None
