# EpyNN/nnlibs/rnn/backward.py
# Related third party imports
import numpy as np


def initialize_backward(layer, dX):
    """Backward cache initialization.

    :param layer: An instance of RNN layer.
    :type layer: :class:`nnlibs.rnn.models.RNN`

    :param dX: Output of backward propagation from next layer.
    :type dX: :class:`numpy.ndarray`

    :return: Input of backward propagation for current layer.
    :rtype: :class:`numpy.ndarray`

    :return: Next cell state initialized with zeros.
    :rtype: :class:`numpy.ndarray`
    """
    dA = layer.bc['dA'] = dX if layer.sequences else np.zeros(layer.fs['h'])

    if not layer.sequences:
        dA[:, -1] = dX

    cache_keys = ['dh', 'dhn']

    layer.bc.update({k: np.zeros(layer.fs['h']) for k in cache_keys})

    layer.bc['dX'] = np.zeros(layer.fs['X'])

    dhn = layer.bc['dhn'][:, 0]

    return dA, dhn


def rnn_backward(layer, dX):
    """Backward propagate error through RNN cells to previous layer.
    """
    # (1) Initialize cache and hidden cell state gradients
    dA, dhn = initialize_backward(layer, dX)

    # Iterate over reversed sequence steps
    for s in reversed(range(layer.d['s'])):

        # (2s) Slice sequence (m, s, h) with respect to step
        dA = layer.bc['dA'][:, s]

        # (3s) Gradient of the loss with respect to hidden cell state
        dh = dA + dhn
        dh = layer.bc['dh'][:, s] = dh * layer.activate(layer.fc['h'][:, s], linear=False, deriv=True)

        # (4s) Gradient of the loss with respect to next hidden state at s-1
        dhn = layer.bc['dhn'][:, s] = np.dot(dh, layer.p['W'].T)

        # (5s) Gradient of the loss with respect to X
        layer.bc['dX'][:, s] = np.dot(dh, layer.p['U'].T)

    dX = layer.bc['dX']

    return dX    # To previous layer
