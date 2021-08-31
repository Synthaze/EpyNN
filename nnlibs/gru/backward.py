# EpyNN/nnlibs/gru/backward.py
# Related third party imports
import numpy as np


def initialize_backward(layer, dX):
    """Backward cache initialization.

    :param layer: An instance of GRU layer.
    :type layer: :class:`nnlibs.gru.models.GRU`

    :param dX: Output of backward propagation from next layer.
    :type dX: :class:`numpy.ndarray`

    :return: Input of backward propagation for current layer.
    :rtype: :class:`numpy.ndarray`

    :return: Next cell state initialized with zeros.
    :rtype: :class:`numpy.ndarray`
    """
    if layer.sequences:
        dA = dX                         # Full length sequence
    elif not layer.sequences:
        dA = np.zeros(layer.fs['h'])    # Empty full length sequence
        dA[:, -1] = dX                  # Assign to last index

    cache_keys = ['dh', 'dhh', 'dz', 'dr', 'dhn']
    layer.bc.update({k: np.zeros(layer.fs['h']) for k in cache_keys})

    layer.bc['dA'] = dA
    layer.bc['dX'] = np.zeros(layer.fs['X'])    # To previous layer

    dhn = layer.bc['dhn'][:, 0]                 # To previous cell

    return dA, dhn


def gru_backward(layer, dX):
    """Backward propagate error gradients to previous layer.
    """
    # (1) Initialize cache and hidden cell state gradients
    dA, dhn = initialize_backward(layer, dX)

    # Reverse iteration over sequence steps
    for s in reversed(range(layer.d['s'])):

        # (2s) Slice sequence (m, s, u) with respect to step
        dA = layer.bc['dA'][:, s]

        # (3s) Gradient of the loss with respect to hidden cell state
        dh = layer.bc['dh'][:, s] = (
            (dA + dhn)
        )

        # (4s) Gradient of the loss w.r.t hidden hat (hh)
        dhh = layer.bc['dhh'][:, s] = (
            dh
            * (1-layer.fc['z'][:, s])
            * layer.activate(layer.fc['hh_'][:, s], deriv=True)
        )

        # (5s) Gradient of the loss w.r.t update gate
        dz = layer.bc['dz'][:, s] = (
            dh
            * (layer.fc['hp'][:, s]-layer.fc['hh'][:, s])
            * layer.activate_update(layer.fc['z_'][:, s], deriv=True)
        )

        # (6s) Gradient of the loss w.r.t reset gate
        dr = layer.bc['dr'][:, s] = (
            np.dot(dhh, layer.p['Wh'].T)
            * layer.fc['hp'][:, s]
            * layer.activate_reset(layer.fc['r_'][:, s], deriv=True)
        )

        # (7s) Gradient of the loss w.r.t previous hidden state
        dhn = layer.bc['dhn'][:, s] = (
            np.dot(dhh, layer.p['Wh'].T) * layer.fc['r'][:, s]
            + np.dot(dz, layer.p['Wz'].T)
            + dh * layer.fc['z'][:, s]
            + np.dot(dr, layer.p['Wr'].T)
        )

        # (8s) Gradient of the loss w.r.t to X
        dX = layer.bc['dX'][:, s] = (
            np.dot(dr, layer.p['Ur'].T)
            + np.dot(dz, layer.p['Uz'].T)
            + np.dot(dhh, layer.p['Uh'].T)
        )

    dX = layer.bc['dX']

    return dX    # To previous layer
