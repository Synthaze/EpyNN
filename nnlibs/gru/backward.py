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

    cache_keys = ['dh', 'dhh', 'dz', 'dr', 'dhfhh', 'dhfr', 'dhfzi', 'dhfz', 'dhn']
    layer.bc.update({k: np.zeros(layer.fs['h']) for k in cache_keys})

    layer.bc['dA'] = dA
    layer.bc['dX'] = np.zeros(layer.fs['X'])    # To previous layer

    dhn = layer.bc['dhn'][:, 0]                 # To previous cell

    return dA, dhn


def gru_backward(layer, dX):
    """Backward propagate error gradients through GRU cells to previous layer.
    """
    # (1) Initialize cache and hidden cell state gradients
    dA, dhn = initialize_backward(layer, dX)

    # Reverse iteration over sequence steps
    for s in reversed(range(layer.d['s'])):

        # (2s) Slice sequence (m, s, u) with respect to step
        dA = layer.bc['dA'][:, s]

        # (3s) Gradient of the loss with respect to hidden cell state
        dh = dA      # Grad. from next layer (dA)
        dh += dhn    # Grad. from next cell (dhn)

        # (4s) Gradient of the loss w.r.t hidden hat (hh)
        dhh = dh * (1-layer.fc['z'][:, s])
        dhh = layer.bc['dhh'][:, s] = dhh * layer.activate(layer.fc['hh'][:, s], linear=False, deriv=True)

        # (5s) Gradient of the loss w.r.t update gate
        dz = dh * (layer.fc['hp'][:, s] - layer.fc['hh'][:, s])
        dz = layer.bc['dz'][:, s] = dz * layer.activate_update(layer.fc['z'][:, s], linear=False, deriv=True)

        # (6s) Gradient of the loss w.r.t reset gate
        dr = np.dot(dhh, layer.p['Wh'].T)
        dr = dr * layer.fc['hp'][:, s]
        dr = layer.bc['dr'][:, s] = dr * layer.activate_reset(layer.fc['r'][:, s], linear=False, deriv=True)

        # (7s) Gradient of the loss w.r.t previous hidden state
        dhfhh = np.dot(dhh, layer.p['Wh'].T)
        dhfhh = layer.bc['dhfhh'][:, s] = dhfhh * layer.fc['r'][:, s]
        dhfzi = layer.bc['dhfzi'][:, s] = np.dot(dz, layer.p['Wz'].T)
        dhfz = layer.bc['dhfz'][:, s] = dh * layer.fc['z'][:, s]
        dhfr = layer.bc['dhfr'][:, s] = np.dot(dr, layer.p['Wr'].T)

        dhn = layer.bc['dhn'][:, s] = dhfhh + dhfr + dhfzi + dhfz    # To previous cell

        # (8s) Gradient of the loss w.r.t to X
        dX = np.dot(dr, layer.p['Ur'].T)
        dX += np.dot(dz, layer.p['Uz'].T)
        dX += np.dot(dhh, layer.p['Uh'].T)
        layer.bc['dX'][:, s] = dX

    dX = layer.bc['dX']

    return dX    # To previous layer
