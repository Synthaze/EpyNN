# EpyNN/nnlibs/gru/parameters.py
# Related third party imports
import numpy as np


def gru_compute_shapes(layer, A):
    """Compute shapes for GRU layer object

    :param layer: An instance of the :class:`nnlibs.gru.models.GRU`
    :type layer: class:`nnlibs.gru.models.GRU`
    """

    X = A

     # (s, v, m)
    layer.d['s'] = X.shape[0]    # Length of sequence
    layer.d['v'] = X.shape[1]    # Vocabulary size
    layer.d['m'] = X.shape[2]    # Number of samples
    # Output length
    layer.d['o'] = 2 if layer.binary else layer.d['v']

    hv = (layer.d['h'], layer.d['v'])
    hh = (layer.d['h'], layer.d['h'])
    h1 = (layer.d['h'], 1)
    oh = (layer.d['o'], layer.d['h'])
    o1 = (layer.d['o'], 1)

    layer.fs['Wz'] = layer.fs['Wr'] = layer.fs['Wh'] = hv
    layer.fs['Uz'] = layer.fs['Ur'] = layer.fs['Uh'] = hh
    layer.fs['bz'] = layer.fs['br'] = layer.fs['bh'] = h1

    layer.fs['W'] = oh
    layer.fs['b'] = o1

    hhm = layer.fs['h'] = layer.fs['C'] = (layer.d['h'], layer.d['h'], layer.d['m'])
    hom = layer.fs['A'] = (layer.d['h'], layer.d['o'], layer.d['m'])

    # Shapes to initialize backward cache
    hvm = layer.fs['X'] = layer.bs['dA'] = (max(layer.d['h'],layer.d['s']), layer.d['v'], layer.d['m'])

    return None


def gru_initialize_parameters(layer):
    """Dummy function - Initialize parameters for GRU layer object

    :param layer: An instance of the :class:`nnlibs.gru.models.GRU`
    :type layer: class:`nnlibs.gru.models.GRU`
    """

    # Init W, b - Forget gate
    layer.p['Wz'] = layer.initialization(layer.fs['Wz'], rng=layer.np_rng)
    layer.p['Uz'] = layer.initialization(layer.fs['Uz'], rng=layer.np_rng)
    layer.p['bz'] = np.zeros(layer.fs['bz'])
    # Init W, b - Forget gate
    layer.p['Wr'] = layer.initialization(layer.fs['Wr'], rng=layer.np_rng)
    layer.p['Ur'] = layer.initialization(layer.fs['Ur'], rng=layer.np_rng)
    layer.p['br'] = np.zeros(layer.fs['br'])
    # Init W, b - Forget gate
    layer.p['Wh'] = layer.initialization(layer.fs['Wh'], rng=layer.np_rng)
    layer.p['Uh'] = layer.initialization(layer.fs['Uh'], rng=layer.np_rng)
    layer.p['bh'] = np.zeros(layer.fs['bh'])
    # Init W, b - Forget gate
    layer.p['W'] = layer.initialization(layer.fs['W'])
    layer.p['b'] = np.zeros(layer.fs['b'])

    return None


def gru_compute_gradients(layer):
    """Dummy function - Update weight and bias gradients for GRU layer object

    :param layer: An instance of the :class:`nnlibs.gru.models.GRU`
    :type layer: class:`nnlibs.gru.models.GRU`
    """

    for parameter in layer.p.keys():
        gradient = 'd' + parameter
        layer.g[gradient] = np.zeros_like(layer.p[parameter])

    for s in reversed(range(layer.d['h'])):

        h = layer.fc['h'][s]
        hp = layer.fc['h'][s-1]

        X = layer.fc['X'][s]

        dX = layer.bc['dX'][s] if not layer.binary else layer.bc['dX']

        # Retrieve dv and update dWv and dbv
        layer.g['dW'] += 1./ layer.d['m'] * np.dot(dX, h.T)
        layer.g['db'] += 1./ layer.d['m'] * np.sum(dX, axis=1, keepdims=True)

        # Retrieve dv and update dWv and dbv
        dhh = layer.bc['dhh'][s]
        layer.g['dWh'] += 1./ layer.d['m'] * np.dot(dhh, X.T)
        layer.g['dUh'] += 1./ layer.d['m'] * np.dot(dhh, (layer.fc['r'][s] * hp).T)
        layer.g['dbh'] += 1./ layer.d['m'] * np.sum(dhh, axis=1, keepdims=True)
        # Retrieve dv and update dWv and dbv
        dr = layer.bc['dr'][s]
        layer.g['dWr'] += 1./ layer.d['m'] * np.dot(dr, X.T)
        layer.g['dUr'] += 1./ layer.d['m'] * np.dot(dr, hp.T)
        layer.g['dbr'] += 1./ layer.d['m'] * np.sum(dr, axis=1, keepdims=True)
        # Retrieve dv and update dWv and dbv
        dz = layer.bc['dz'][s]
        layer.g['dWz'] += 1./ layer.d['m'] * np.dot(dz, X.T)
        layer.g['dUz'] += 1./ layer.d['m'] * np.dot(dz, hp.T)
        layer.g['dbz'] += 1./ layer.d['m'] * np.sum(dz, axis=1, keepdims=True)

    return None


def gru_update_parameters(layer):
    """Dummy function - Update parameters for GRU layer object

    :param layer: An instance of the :class:`nnlibs.gru.models.GRU`
    :type layer: class:`nnlibs.gru.models.GRU`
    """

    for gradient in layer.g.keys():

        parameter = gradient[1:]

        layer.p[parameter] -= layer.lrate[layer.e] * layer.g[gradient]

    return None
