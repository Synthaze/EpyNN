# EpyNN/nnlibs/lstm/parameters.py
# Related third party imports
import numpy as np


def lstm_compute_shapes(layer, A):
    """Compute shapes for LSTM layer object

    :param layer: An instance of the :class:`nnlibs.lstm.models.LSTM`
    :type layer: class:`nnlibs.lstm.models.LSTM`
    """

    X = A

    layer.fs['X'] = X.shape

    layer.d['v'] = layer.fs['X'][0]
    layer.d['t'] = layer.fs['X'][1]
    layer.d['m'] = layer.fs['X'][2]

    layer.d['z'] = layer.d['h'] + layer.d['v']

    if layer.binary == False:
        layer.d['o'] = layer.d['v']
    else:
        layer.d['o'] = 2

    h1 = (layer.d['h'], 1)
    hz = (layer.d['h'], layer.d['z'])
    oh = (layer.d['o'], layer.d['h'])
    o1 = (layer.d['o'], 1)

    layer.fs['Wf'] = layer.fs['Wi'] = layer.fs['Wg'] = layer.fs['Wo'] = hz
    layer.fs['bf'] = layer.fs['bi'] = layer.fs['bg'] = layer.fs['bo'] = h1

    layer.fs['W'] = oh
    layer.fs['b'] = o1

    tvm = layer.fs['Xt'] = (layer.d['t'], layer.d['v'], layer.d['m'])
    thm = layer.fs['h'] = layer.fs['C'] = (layer.d['t'], layer.d['h'], layer.d['m'])
    tom = layer.fs['A'] = (layer.d['t'], layer.d['o'], layer.d['m'])

    tzm = layer.fs['z'] = (layer.d['t'], layer.d['z'], layer.d['m'])

    hm = layer.fs['Ct'] = layer.fs['ht'] = (layer.d['h'], layer.d['m'])

    return None


def lstm_initialize_parameters(layer):
    """Dummy function - Initialize parameters for LSTM layer object

    :param layer: An instance of the :class:`nnlibs.lstm.models.LSTM`
    :type layer: class:`nnlibs.lstm.models.LSTM`
    """

    # Init W, b - Forget gate
    layer.p['Wf'] = layer.initialization(layer.fs['Wf'], rng=layer.np_rng)
    layer.p['bf'] = np.zeros(layer.fs['bf'])
    # Init W, b - Forget gate
    layer.p['Wi'] = layer.initialization(layer.fs['Wi'], rng=layer.np_rng)
    layer.p['bi'] = np.zeros(layer.fs['bi'])
    # Init W, b - Forget gate
    layer.p['Wg'] = layer.initialization(layer.fs['Wg'], rng=layer.np_rng)
    layer.p['bg'] = np.zeros(layer.fs['bg'])
    # Init W, b - Forget gate
    layer.p['Wo'] = layer.initialization(layer.fs['Wo'], rng=layer.np_rng)
    layer.p['bo'] = np.zeros(layer.fs['bo'])

    # W, U, b - _ gate
    layer.p['W'] = layer.initialization(layer.fs['W'], rng=layer.np_rng)
    layer.p['b'] = np.zeros(layer.fs['b'])

    return None


def lstm_update_gradients(layer):
    """Dummy function - Update weight and bias gradients for LSTM layer object

    :param layer: An instance of the :class:`nnlibs.lstm.models.LSTM`
    :type layer: class:`nnlibs.lstm.models.LSTM`
    """

    for parameter in layer.p.keys():
        gradient = 'd' + parameter
        layer.g[gradient] = np.zeros_like(layer.p[parameter])

    for t in reversed(range(layer.d['t'])):

        h = layer.fc['h'][t]

        z = layer.fc['z'][t]

        if layer.binary == False:
            dXt = layer.bc['dXt'][t]
        else:
            dXt = layer.bc['dXt']

        layer.g['dW'] += 1./ layer.d['m'] * np.dot(dXt, h.T)
        layer.g['db'] += 1./ layer.d['m'] * np.sum(dXt, axis=1, keepdims=True)

        # Retrieve do and update dWo and dbo
        do = layer.bc['do'][t]
        layer.g['dWo'] += 1./ layer.d['m'] * np.dot(do, z.T)
        layer.g['dbo'] += 1./ layer.d['m'] * np.sum(do, axis=1, keepdims=True)
        # Retrieve dg and update dWg and dbg
        dg = layer.bc['dg'][t]
        layer.g['dWg'] += 1./ layer.d['m'] * np.dot(dg, z.T)
        layer.g['dbg'] += 1./ layer.d['m'] * np.sum(dg, axis=1, keepdims=True)
        # Retrieve di and update dWi and dbi
        di = layer.bc['di'][t]
        layer.g['dWi'] += 1./ layer.d['m'] * np.dot(di, z.T)
        layer.g['dbi'] += 1./ layer.d['m'] * np.sum(di, axis=1, keepdims=True)
        # Retrieve df and update dWf and dbf
        df = layer.bc['df'][t]
        layer.g['dWf'] += 1./ layer.d['m'] * np.dot(df, z.T)
        layer.g['dbf'] += 1./ layer.d['m'] * np.sum(df, axis=1, keepdims=True)

    return None


def lstm_update_parameters(layer):
    """Dummy function - Update parameters for LSTM layer object

    :param layer: An instance of the :class:`nnlibs.lstm.models.LSTM`
    :type layer: class:`nnlibs.lstm.models.LSTM`
    """

    for gradient in layer.g.keys():

        parameter = gradient[1:]

        layer.p[parameter] -= layer.lrate[layer.e] * layer.g[gradient]

    return None
