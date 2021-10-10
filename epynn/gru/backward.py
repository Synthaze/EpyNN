# EpyNN/epynn/gru/backward.py
# Related third party imports
import numpy as np


def initialize_backward(layer, dX):
    """Backward cache initialization.

    :param layer: An instance of GRU layer.
    :type layer: :class:`epynn.gru.models.GRU`

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

    cache_keys = ['dh_', 'dh', 'dhn', 'dhh_', 'dz_', 'dr_']
    layer.bc.update({k: np.zeros(layer.fs['h']) for k in cache_keys})

    layer.bc['dA'] = dA
    layer.bc['dX'] = np.zeros(layer.fs['X'])    # To previous layer

    dh = layer.bc['dh'][:, 0]                   # To previous step

    return dA, dh


def gru_backward(layer, dX):
    """Backward propagate error gradients to previous layer.
    """
    # (1) Initialize cache and hidden state gradients
    dA, dh = initialize_backward(layer, dX)

    # Reverse iteration over sequence steps
    for s in reversed(range(layer.d['s'])):

        # (2s) Slice sequence (m, s, u) w.r.t step
        dA = layer.bc['dA'][:, s]          # dL/dA

        # (3s) Retrieve previous hidden state
        dhn = layer.bc['dhn'][:, s] = dh   # dL/dhn

        # (4s) Gradient of the loss w.r.t hidden state h_
        dh_ = layer.bc['dh_'][:, s] = (
            (dA + dhn)
        )   # dL/dh_

        # (5s) Gradient of the loss w.r.t hidden hat hh_
        dhh_ = layer.bc['dhh_'][:, s] = (
            dh_
            * (1-layer.fc['z'][:, s])
            * layer.activate(layer.fc['hh_'][:, s], deriv=True)
        )   # dL/dhh_

        # (6s) Gradient of the loss w.r.t update gate z_
        dz_ = layer.bc['dz_'][:, s] = (
            dh_
            * (layer.fc['hp'][:, s]-layer.fc['hh'][:, s])
            * layer.activate_update(layer.fc['z_'][:, s], deriv=True)
        )   # dL/dz_

        # (7s) Gradient of the loss w.r.t reset gate
        dr_ = layer.bc['dr_'][:, s] = (
            np.dot(dhh_, layer.p['Vhh'].T)
            * layer.fc['hp'][:, s]
            * layer.activate_reset(layer.fc['r_'][:, s], deriv=True)
        )   # dL/dr_

        # (8s) Gradient of the loss w.r.t previous hidden state
        dh = layer.bc['dh'][:, s] = (
            np.dot(dhh_, layer.p['Vhh'].T) * layer.fc['r'][:, s]
            + np.dot(dz_, layer.p['Vz'].T) + dh_ * layer.fc['z'][:, s]
            + np.dot(dr_, layer.p['Vr'].T)
        )   # dL/dh

        # (9s) Gradient of the loss w.r.t to X
        dX = layer.bc['dX'][:, s] = (
            np.dot(dhh_, layer.p['Uhh'].T)
            + np.dot(dz_, layer.p['Uz'].T)
            + np.dot(dr_, layer.p['Ur'].T)
        )   # dL/dX

    dX = layer.bc['dX']

    return dX    # To previous layer
