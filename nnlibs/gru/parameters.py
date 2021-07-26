# EpyNN/nnlibs/gru/parameters.py
# Related third party imports
import numpy as np


def gru_compute_shapes(layer, A):
    """Compute shapes for GRU layer object

    :param layer: An instance of the :class:`nnlibs.gru.models.GRU`
    :type layer: class:`nnlibs.gru.models.GRU`
    """

    X = A

    layer.fs['X'] = X.shape

    layer.d['v'] = layer.fs['X'][0]
    layer.d['t'] = layer.fs['X'][1]
    layer.d['m'] = layer.fs['X'][2]

    if layer.binary == False:
        layer.d['o'] = layer.d['v']
    else:
        layer.d['o'] = 2

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

    tvm = layer.fs['Xt'] = (layer.d['t'], layer.d['v'], layer.d['m'])
    thm = layer.fs['h'] = layer.fs['C'] = (layer.d['t'], layer.d['h'], layer.d['m'])
    tom = layer.fs['A'] = (layer.d['t'], layer.d['o'], layer.d['m'])

    hm = layer.fs['ht'] = (layer.d['h'], layer.d['m'])

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


def gru_update_gradients(layer):
    """Dummy function - Update weight and bias gradients for GRU layer object

    :param layer: An instance of the :class:`nnlibs.gru.models.GRU`
    :type layer: class:`nnlibs.gru.models.GRU`
    """

    for parameter in layer.p.keys():
        gradient = 'd' + parameter
        layer.g[gradient] = np.zeros_like(layer.p[parameter])

    for t in reversed(range(layer.d['t'])):

        h = layer.fc['h'][t]
        hp = layer.fc['h'][t-1]

        Xt = layer.fc['X'][:,t]

        if layer.binary == False:
            dXt = layer.bc['dXt'][t]
        else:
            dXt = layer.bc['dXt']

        # Retrieve dv and update dWv and dbv
        layer.g['dW'] += 1./ layer.d['m'] * np.dot(dXt, h.T)
        layer.g['db'] += 1./ layer.d['m'] * np.sum(dXt, axis=1, keepdims=True)

        # Retrieve dv and update dWv and dbv
        dhh = layer.bc['dhh'][t]
        layer.g['dWh'] += 1./ layer.d['m'] * np.dot(dhh, Xt.T)
        layer.g['dUh'] += 1./ layer.d['m'] * np.dot(dhh, (layer.fc['r'][t] * hp).T)
        layer.g['dbh'] += 1./ layer.d['m'] * np.sum(dhh, axis=1, keepdims=True)
        # Retrieve dv and update dWv and dbv
        dr = layer.bc['dr'][t]
        layer.g['dWr'] += 1./ layer.d['m'] * np.dot(dr, Xt.T)
        layer.g['dUr'] += 1./ layer.d['m'] * np.dot(dr, hp.T)
        layer.g['dbr'] += 1./ layer.d['m'] * np.sum(dr, axis=1, keepdims=True)
        # Retrieve dv and update dWv and dbv
        dz = layer.bc['dz'][t]
        layer.g['dWz'] += 1./ layer.d['m'] * np.dot(dz, Xt.T)
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
