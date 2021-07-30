# EpyNN/nnlibs/rnn/backward.py
# Related third party imports
import numpy as np


def initialize_backward(layer, dA):
    """Backward cache initialization.

    :param layer: RNN layer object
    :type layer: class:`nnlibs.rnn.models.RNN`
    :param dA: Output of backward propagation from next layer
    :type dA: class:`numpy.ndarray`

    :return: Input of backward propagation for current layer
    :rtype: class:`numpy.ndarray`
    :return: Next cell state initialized with zeros.
    :rtype: class:`numpy.ndarray`
    """
    dX = layer.bc['dX'] = dA

    layer.bc['dh'] = np.zeros(layer.fs['h'])
    layer.bc['dhn'] = np.zeros_like(layer.bc['dh'])
    layer.bc['dA'] = np.zeros(layer.fs['X'])

    dhn = layer.bc['dhn'][0]

    return dX, dhn


def rnn_backward(layer, dA):
    """Backward propagate signal through RNN cells to previous layer.
    """
    # (1) Initialize cache and cell state
    dX, dhn = initialize_backward(layer, dA)

    # Iterate through reversed sequence to previous cell
    for s in reversed(range(layer.d['h'])):

        # (2s) Slice sequence (l, v, m) with respect to step
        dX = dX if layer.binary else layer.bc['dX'][s]

        # (3s) Compute partial derivative of cell state error
        dh = np.zeros_like(dhn) if layer.binary else dhn
        dh += np.dot(layer.p['W'].T, dX)
        dh = layer.bc['dh'][s] = dh * layer.activate_hidden(layer.fc['h'][s], deriv=True)

        # (4s) With respect to cell state
        dhn = layer.bc['dhn'][s] = np.dot(layer.p['Wh'].T, dh)

        # (5s) With respect to cell input
        dA = layer.bc['dA'][s] = np.dot(layer.p['Wx'].T, dh)

    dA = layer.bc['dA']

    return dA    # To previous layer
