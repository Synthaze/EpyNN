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

    cache_keys = ['dh', 'dhn']

    layer.bc.update({k: np.zeros(layer.fs['h']) for k in cache_keys})

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

        # (3s)
        dh = dX + dhn
        dh = layer.bc['dh'][:, s] = layer.activate(layer.fc['h'][:, s]**2, deriv=True) * dh

        # (4s)
        dhn = layer.bc['dhn'][:, s] = np.dot(dh, layer.p['W'].T)

        # (5s)
        layer.bc['dA'][:, s] = np.dot(dh, layer.p['U'].T)

        #
        if not layer.sequences: break

    dA = layer.bc['dA']

    return dA    # To previous layer
