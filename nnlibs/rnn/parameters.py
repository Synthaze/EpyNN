# EpyNN/nnlibs/rnn/parameters.py
# Related third party imports
import numpy as np


def rnn_compute_shapes(layer, A):
    """Compute shapes for RNN layer object

    :param layer: An instance of the :class:`nnlibs.rnn.models.RNN`
    :type layer: class:`nnlibs.rnn.models.RNN`
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

    hv = layer.fs['Uh'] = ( layer.d['h'], layer.d['v'] )
    hh = layer.fs['Vh'] = ( layer.d['h'], layer.d['h'] )
    h1 = layer.fs['bh'] = ( layer.d['h'], 1 )

    oh = layer.fs['W'] = ( layer.d['o'], layer.d['h'] )
    o1 = layer.fs['b'] = ( layer.d['o'], 1 )

    tvm = layer.fs['Xt'] = (layer.d['t'], layer.d['v'], layer.d['m'])
    thm = layer.fs['h'] = (layer.d['t'], layer.d['h'], layer.d['m'])
    tom = layer.fs['A'] = (layer.d['t'], layer.d['o'], layer.d['m'])

    hm = layer.fs['ht'] = (layer.d['h'], layer.d['m'])

    return None


def rnn_initialize_parameters(layer):
    """Dummy function - Initialize parameters for RNN layer object

    :param layer: An instance of the :class:`nnlibs.rnn.models.RNN`
    :type layer: class:`nnlibs.rnn.models.RNN`
    """

    # W, U, b - _ gate
    layer.p['Uh'] = layer.initialization(layer.fs['Uh'], rng=layer.np_rng)
    layer.p['Vh'] = layer.initialization(layer.fs['Vh'], rng=layer.np_rng)
    layer.p['bh'] = np.zeros(layer.fs['bh'])

    # W, U, b - _ gate
    layer.p['W'] = layer.initialization(layer.fs['W'], rng=layer.np_rng)
    layer.p['b'] = np.zeros(layer.fs['b'])

    return None


def rnn_update_gradients(layer):
    """Dummy function - Update weight and bias gradients for RNN layer object

    :param layer: An instance of the :class:`nnlibs.rnn.models.RNN`
    :type layer: class:`nnlibs.rnn.models.RNN`
    """

    for parameter in layer.p.keys():
        gradient = 'd' + parameter
        layer.g[gradient] = np.zeros_like(layer.p[parameter])

    for t in reversed(range(layer.d['t'])):

        h = layer.fc['h'][t]
        hp = layer.fc['h'][t-1]

        Xt = layer.fc['X'][:, t]

        # _

        if layer.binary == False:
            dXt = layer.bc['dXt'][t]
        else:
            dXt = layer.bc['dXt']

        layer.g['dW'] += 1./ layer.d['m'] * np.dot(dXt, h.T)
        layer.g['db'] += 1./ layer.d['m'] * np.sum(dXt, axis=1, keepdims=True)

        # _
        _dh = layer.bc['_dh']
        layer.g['dUh'] += 1./ layer.d['m'] * np.dot(_dh, Xt.T)
        layer.g['dVh'] += 1./ layer.d['m'] * np.dot(_dh, hp.T)
        layer.g['dbh'] += 1./ layer.d['m'] * np.sum(_dh, axis=1, keepdims=True)

    return None


def rnn_update_parameters(layer):
    """Dummy function - Update parameters for RNN layer object

    :param layer: An instance of the :class:`nnlibs.rnn.models.RNN`
    :type layer: class:`nnlibs.rnn.models.RNN`
    """

    for gradient in layer.g.keys():

        parameter = gradient[1:]

        layer.p[parameter] -= layer.lrate[layer.e] * layer.g[gradient]

    return None
