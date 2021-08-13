# EpyNN/nnlibs/lstm/backward.py
# Related third party imports
import numpy as np


def initialize_backward(layer, dA):
    """Backward cache initialization.

    :param layer: An instance of LSTM layer.
    :type layer: :class:`nnlibs.lstm.models.LSTM`

    :param dA: Output of backward propagation from next layer
    :type dA: :class:`numpy.ndarray`

    :return: Input of backward propagation for current layer
    :rtype: :class:`numpy.ndarray`

    :return: Next cell state initialized with zeros.
    :rtype: :class:`numpy.ndarray`

    :return: Next memory state initialized with zeros.
    :rtype: :class:`numpy.ndarray`
    """
    dX = layer.bc['dX'] = dA

    cache_keys = ['dh', 'do', 'dg', 'di', 'df', 'dz', 'dC', 'dCn', 'dhn']

    layer.bc.update({k: np.zeros(layer.fs['h']) for k in cache_keys})

    layer.bc['dA'] = np.zeros(layer.fs['X'])

    dhn = np.zeros_like(layer.bc['dh'][:, 0])
    dCn = np.zeros_like(layer.bc['dC'][:, 0])

    return dX, dhn, dCn


def lstm_backward(layer, dA):
    """Backward propagate signal through LSTM cells to previous layer.
    """
    dX, dhn, dCn = initialize_backward(layer, dA)

    # Loop through steps
    for s in reversed(range(layer.d['s'])):

        # (2s)
        dX = layer.bc['dX'][:, s] if layer.sequences else dX

        # (3s)
        dh = dX + dhn

        # (4s)
        do = dh * layer.activate(layer.fc['C'][:, s])
        do = layer.bc['do'][:, s] = do * layer.activate_output(layer.fc['o'][:, s], deriv=True)

        # (5s)
        dC = layer.fc['o'][:, s] * dh * layer.activate(layer.activate(layer.fc['C'][:, s]), deriv=True)
        dC =  layer.bc['dC'][:, s] = dC + dCn

        # (6.1s)
        dg = dC * layer.fc['i'][:, s]
        dg = layer.bc['dg'][:, s] = dg * layer.activate_candidate(layer.fc['g'][:, s], deriv=True)

        # (6.2s)
        di = dC * layer.fc['g'][:, s]
        di = layer.bc['di'][:, s] = di * layer.activate_input(layer.fc['i'][:, s], deriv=True)

        # (7s)
        df = dC * layer.fc['C'][:, s - 1]
        df = layer.bc['df'][:, s] = df * layer.activate_forget(layer.fc['f'][:, s], deriv=True)

        # (8s)
        dz = np.dot(do, layer.p['Wo'].T)
        dz += np.dot(dg, layer.p['Wg'].T)
        dz += np.dot(di, layer.p['Wi'].T)
        dz += np.dot(df, layer.p['Wf'].T)
        layer.bc['dz'][:, s] = dz

        dhn = layer.bc['dhn'][:, s] = dz[:, :layer.d['h']]

        # (9s)
        dCn = layer.bc['dCn'][:, s] = layer.fc['f'][:, s] * dC

        # (10s)
        dA = np.dot(dg, layer.p['Ug'].T)
        dA += np.dot(do, layer.p['Uo'].T)
        dA += np.dot(di, layer.p['Ui'].T)
        dA += np.dot(df, layer.p['Uf'].T)
        layer.bc['dA'][:, s] = dA

        #
        if not layer.sequences: break

    dA = layer.bc['dA']

    return dA    # To previous layer
