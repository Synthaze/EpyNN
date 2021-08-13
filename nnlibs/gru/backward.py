# EpyNN/nnlibs/gru/backward.py
# Related third party imports
import numpy as np


def initialize_backward(layer, dA):
    """Backward cache initialization.

    :param layer: An instance of GRU layer.
    :type layer: :class:`nnlibs.gru.models.GRU`

    :param dA: Output of backward propagation from next layer
    :type dA: :class:`numpy.ndarray`

    :return: Input of backward propagation for current layer
    :rtype: :class:`numpy.ndarray`

    :return: Next cell state initialized with zeros.
    :rtype: :class:`numpy.ndarray`
    """
    dX = layer.bc['dX'] = dA

    cache_keys = ['dh', 'dhh', 'dz', 'dr', 'dhfhh', 'dhfr', 'dhfzi', 'dhfz', 'dhn']

    layer.bc.update({k: np.zeros(layer.fs['h']) for k in cache_keys})

    layer.bc['dA'] = np.zeros(layer.fs['X'])

    dhn = np.zeros_like(layer.bc['dh'][:, 0])

    return dX, dhn


def gru_backward(layer, dA):
    """Backward propagate signal through GRU cells to previous layer.
    """
    # (1)
    dX, dhn = initialize_backward(layer, dA)

    # Loop through steps
    for s in reversed(range(layer.d['s'])):

        # (2s)
        dX = layer.bc['dX'][:, s] if layer.sequences else dX

        # (3s)
        dh =  dX + dhn

        # (4s)
        dhh = dh * (1-layer.fc['z'][:, s])
        dhh = layer.bc['dhh'][:, s] = dhh * layer.activate(layer.fc['hh'][:, s], deriv=True)

        # (5s)
        dz = dh * (layer.fc['h'][:, s - 1] - layer.fc['hh'][:, s])
        dz = layer.bc['dz'][:, s] = dz * layer.activate_update(layer.fc['z'][:, s], deriv=True)

        # (6s)
        dr = np.dot(dhh, layer.p['Wh'].T)
        dr = dr * layer.fc['h'][:, s - 1]
        dr = layer.bc['dr'][:, s] = dr * layer.activate_reset(layer.fc['r'][:, s], deriv=True)

        # (7s)
        dhfhh = np.dot(dhh, layer.p['Wh'].T)
        dhfhh = layer.bc['dhfhh'][:, s] = dhfhh * layer.fc['r'][:, s]
        dhfzi = layer.bc['dhfzi'][:, s] = np.dot(dz, layer.p['Wz'].T)
        dhfz = layer.bc['dhfz'][:, s] = dh * layer.fc['z'][:, s]
        dhfr = layer.bc['dhfr'][:, s] = np.dot(dr, layer.p['Wr'].T)

        dhn = layer.bc['dhn'][:, s] = dhfhh + dhfr + dhfzi + dhfz

        # (8s)
        dA = np.dot(dr, layer.p['Ur'].T)
        dA += np.dot(dz, layer.p['Uz'].T)
        dA += np.dot(dhh, layer.p['Uh'].T)
        layer.bc['dA'][:, s] = dA

        #
        if not layer.sequences: break

    dA = layer.bc['dA']

    return dA    # To previous layer
