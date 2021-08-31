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

    :return: Next hidden cell state initialized with zeros.
    :rtype: :class:`numpy.ndarray`
    """
    if layer.sequences:
        dA = dX                         # Full length sequence
    elif not layer.sequences:
        dA = np.zeros(layer.fs['h'])    # Empty full length sequence
        dA[:, -1] = dX                  # Assign to last index

    cache_keys = ['dh', 'dhn']
    layer.bc.update({k: np.zeros(layer.fs['h']) for k in cache_keys})

    layer.bc['dA'] = dA
    layer.bc['dX'] = np.zeros(layer.fs['X'])    # To previous layer

    dhn = layer.bc['dhn'][:, 0]                 # To previous cell

    return dA, dhn


def rnn_backward(layer, dX):
    """Backward propagate error gradients to previous layer.
    """
    # (1) Initialize cache and hidden cell state gradient
    dA, dhn = initialize_backward(layer, dX)

    # Reverse iteration over sequence steps
    for s in reversed(range(layer.d['s'])):

        # (2s) Slice sequence (m, s, u) with respect to step
        dA = layer.bc['dA'][:, s]

        # (3s) Gradient of the loss with respect to hidden cell state
        dh = layer.bc['dh'][:, s] = (
            (dA + dhn)
            * layer.activate(layer.fc['h_'][:, s], deriv=True)
        )

        # (4s) Gradient of the loss w.r.t previous hidden state
        dhn = np.dot(dh, layer.p['W'].T)

        # (5s) Gradient of the loss with respect to X
        layer.bc['dX'][:, s] = np.dot(dh, layer.p['U'].T)

    dX = layer.bc['dX']

    return dX    # To previous layer
