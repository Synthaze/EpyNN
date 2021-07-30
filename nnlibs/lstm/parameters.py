# EpyNN/nnlibs/lstm/parameters.py
# Related third party imports
import numpy as np


def lstm_compute_shapes(layer, A):
    """Compute shapes for LSTM layer object

    :param layer: An instance of the :class:`nnlibs.lstm.models.LSTM`
    :type layer: class:`nnlibs.lstm.models.LSTM`
    """
    X = A    # Input of current layer

     # (s, v, m)
    layer.d['s'] = X.shape[0]    # Length of sequence
    layer.d['v'] = X.shape[1]    # Vocabulary size
    layer.d['m'] = X.shape[2]    # Number of samples
    # Output length
    layer.d['o'] = 2 if layer.binary else layer.d['v']

    layer.d['z'] = layer.d['h'] + layer.d['v']

    hz = (layer.d['h'], layer.d['z'])
    h1 = (layer.d['h'], 1)

    oh = (layer.d['o'], layer.d['h'])
    o1 = (layer.d['o'], 1)

    layer.fs['Wf'] = layer.fs['Wi'] = layer.fs['Wg'] = layer.fs['Wo'] = hz
    layer.fs['bf'] = layer.fs['bi'] = layer.fs['bg'] = layer.fs['bo'] = h1

    layer.fs['W'] = oh
    layer.fs['b'] = o1

    # Shapes to initialize forward cache
    hhm = layer.fs['h'] = layer.fs['C'] = (layer.d['h'], layer.d['h'], layer.d['m'])
    ohm = layer.fs['A'] = (layer.d['h'], layer.d['o'], layer.d['m'])
    # Shapes to initialize backward cache
    hvm = layer.fs['X'] = layer.bs['dA'] = (max(layer.d['h'],layer.d['s']), layer.d['v'], layer.d['m'])
    hzm = layer.fs['z'] = (layer.d['h'], layer.d['z'], layer.d['m'])


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


def lstm_compute_gradients(layer):
    """Dummy function - Update weight and bias gradients for LSTM layer object

    :param layer: An instance of the :class:`nnlibs.lstm.models.LSTM`
    :type layer: class:`nnlibs.lstm.models.LSTM`
    """

    for parameter in layer.p.keys():
        gradient = 'd' + parameter
        layer.g[gradient] = np.zeros_like(layer.p[parameter])

    for s in reversed(range(layer.d['h'])):

        h = layer.fc['h'][s]
        z = layer.fc['z'][s]

        dX = layer.bc['dX'][s] if not layer.binary else layer.bc['dX']
        layer.g['dW'] += 1./ layer.d['m'] * np.dot(dX, h.T)
        layer.g['db'] += 1./ layer.d['m'] * np.sum(dX, axis=1, keepdims=True)

        # Retrieve do and update dWo and dbo
        do = layer.bc['do'][s]
        layer.g['dWo'] += 1./ layer.d['m'] * np.dot(do, z.T)
        layer.g['dbo'] += 1./ layer.d['m'] * np.sum(do, axis=1, keepdims=True)
        # Retrieve dg and update dWg and dbg
        dg = layer.bc['dg'][s]
        layer.g['dWg'] += 1./ layer.d['m'] * np.dot(dg, z.T)
        layer.g['dbg'] += 1./ layer.d['m'] * np.sum(dg, axis=1, keepdims=True)
        # Retrieve di and update dWi and dbi
        di = layer.bc['di'][s]
        layer.g['dWi'] += 1./ layer.d['m'] * np.dot(di, z.T)
        layer.g['dbi'] += 1./ layer.d['m'] * np.sum(di, axis=1, keepdims=True)
        # Retrieve df and update dWf and dbf
        df = layer.bc['df'][s]
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
