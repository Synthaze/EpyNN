# EpyNN/nnlibs/rnn/backward.py
# Related third party imports
import numpy as np


def initialize_backward(layer, dA):
    """Backward cache initialization.

    :param layer: An instance of RNN layer.
    :type layer: :class:`nnlibs.rnn.models.RNN`

    :param dA: Output of backward propagation from next layer
    :type dA: :class:`numpy.ndarray`

    :return: Input of backward propagation for current layer
    :rtype: :class:`numpy.ndarray`

    :return: Next cell state initialized with zeros.
    :rtype: :class:`numpy.ndarray`
    """
    dX = layer.bc['dX'] = dA

    layer.bc['dh'] = np.zeros(layer.fs['h'])
    layer.bc['dhn'] = np.zeros_like(layer.bc['dh'])
    layer.bc['dA'] = np.zeros(layer.fs['X'])

    dhn = layer.bc['dhn'][:, 0]

    return dX, dhn


def rnn_backward(layer, dA):
    """Backward propagate signal through RNN cells to previous layer.
    """
    # (1) Initialize cache and cell state
    dX, dhn = initialize_backward(layer, dA)

    # Iterate through reversed sequence to previous cell
    for s in reversed(range(layer.d['s'])):

        # (2s) Slice sequence (m, s, v) with respect to step
        dX = layer.bc['dX'][:, s] if layer.sequences else dX

        dh = dX + dhn
        
        dh = layer.bc['dh'][:, s] = layer.activate(layer.fc['h'][:, s]**2, deriv=True) * dh

        dhn = layer.bc['dhn'][:, s] = np.dot(dh, layer.p['W'].T)

        layer.bc['dA'][:, s] = np.dot(dh, layer.p['U'].T)

    dA = layer.bc['dA']

    return dA    # To previous layer
