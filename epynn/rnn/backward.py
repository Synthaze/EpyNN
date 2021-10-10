# EpyNN/epynn/rnn/backward.py
# Related third party imports
import numpy as np


def initialize_backward(layer, dX):
    """Backward cache initialization.

    :param layer: An instance of RNN layer.
    :type layer: :class:`epynn.rnn.models.RNN`

    :param dX: Output of backward propagation from next layer.
    :type dX: :class:`numpy.ndarray`

    :return: Input of backward propagation for current layer.
    :rtype: :class:`numpy.ndarray`

    :return: Next hidden state initialized with zeros.
    :rtype: :class:`numpy.ndarray`
    """
    if layer.sequences:
        dA = dX                         # Full length sequence
    elif not layer.sequences:
        dA = np.zeros(layer.fs['h'])    # Empty full length sequence
        dA[:, -1] = dX                  # Assign to last index

    cache_keys = ['dh_', 'dh', 'dhn']
    layer.bc.update({k: np.zeros(layer.fs['h']) for k in cache_keys})

    layer.bc['dA'] = dA
    layer.bc['dX'] = np.zeros(layer.fs['X'])    # To previous layer

    dh = layer.bc['dh'][:, 0]                   # To previous step

    return dA, dh


def rnn_backward(layer, dX):
    """Backward propagate error gradients to previous layer.
    """
    # (1) Initialize cache and hidden state gradient
    dA, dh = initialize_backward(layer, dX)

    # Reverse iteration over sequence steps
    for s in reversed(range(layer.d['s'])):

        # (2s) Slice sequence (m, s, u) w.r.t step
        dA = layer.bc['dA'][:, s]          # dL/dA

        # (3s) Gradient of the loss w.r.t. next hidden state
        dhn = layer.bc['dhn'][:, s] = dh   # dL/dhn

        # (4s) Gradient of the loss w.r.t hidden state h_
        dh_ = layer.bc['dh_'][:, s] = (
            (dA + dhn)
            * layer.activate(layer.fc['h_'][:, s], deriv=True)
        )   # dL/dh_ - To parameters gradients

        # (5s) Gradient of the loss w.r.t hidden state h
        dh = layer.bc['dh'][:, s] = (
            np.dot(dh_, layer.p['V'].T)
        )   # dL/dh - To previous step

        # (6s) Gradient of the loss w.r.t X
        dX = layer.bc['dX'][:, s] = (
            np.dot(dh_, layer.p['U'].T)
        )   # dL/dX - To previous layer

    dX = layer.bc['dX']

    return dX    # To previous layer
