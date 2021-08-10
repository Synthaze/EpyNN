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

    layer.bc['dh'] = np.zeros(layer.fs['h'])
    layer.bc['do'] = np.zeros_like(layer.bc['dh'])
    layer.bc['di'] = np.zeros_like(layer.bc['dh'])
    layer.bc['df'] = np.zeros_like(layer.bc['dh'])
    layer.bc['dg'] = np.zeros_like(layer.bc['dh'])
    layer.bc['dz'] = np.zeros_like(layer.bc['dh'])

    layer.bc['dhn'] = np.zeros(layer.fs['h'])

    layer.bc['dC'] = np.zeros(layer.fs['C'])
    layer.bc['dCn'] = np.zeros(layer.fs['C'])

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

        #
        dX = layer.bc['dX'][:, s] if layer.sequences else dX

        #
        dh = dX + dhn

        #
        do = dh * layer.activate(layer.fc['C'][:, s])
        do = layer.bc['do'][:, s] = do * layer.activate_output(layer.fc['o'][:, s], deriv=True)

        #
        dC = layer.fc['o'][:, s] * dh * layer.activate(layer.activate(layer.fc['C'][:, s]), deriv=True)
        dC =  layer.bc['dC'][:, s] = dC + dCn

        #
        dg = dC * layer.fc['i'][:, s]
        dg = layer.bc['dg'][:, s] = dg * layer.activate_candidate(layer.fc['g'][:, s], deriv=True)

        #
        di = dC * layer.fc['g'][:, s]
        di = layer.bc['di'][:, s] = di * layer.activate_input(layer.fc['i'][:, s], deriv=True)

        #
        df = dC * layer.fc['C'][:, s - 1]
        df = layer.bc['df'][:, s] = df * layer.activate_forget(layer.fc['f'][:, s], deriv=True)

        #
        dz = np.dot(dg, layer.p['Wg'].T)
        dz += np.dot(do, layer.p['Wo'].T)
        dz += np.dot(di, layer.p['Wi'].T)
        dz += np.dot(df, layer.p['Wf'].T)
        dz = layer.bc['dz'] = dz

        #
        dhn = layer.bc['dhn'][:, s] = dz[:, :layer.d['h']]

        #
        dCn = layer.bc['dCn'][:, s] = layer.fc['f'][:, s] * dC

        #
        dA = np.dot(dg, layer.p['Ug'].T)
        dA += np.dot(do, layer.p['Uo'].T)
        dA += np.dot(di, layer.p['Ui'].T)
        dA += np.dot(df, layer.p['Uf'].T)
        layer.bc['dA'][:, s] = dA

    dA = layer.bc['dA']

    return dA    # To previous layer
