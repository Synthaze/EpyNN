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

    layer.bc['dh'] = np.zeros(layer.fs['h'])
    layer.bc['dhh'] = np.zeros(layer.fs['h'])
    layer.bc['dr'] = np.zeros_like(layer.bc['dh'])
    layer.bc['dz'] = np.zeros_like(layer.bc['dh'])

    layer.bc['dhfhh'] = np.zeros_like(layer.bc['dh'])
    layer.bc['dhfr'] = np.zeros_like(layer.bc['dh'])
    layer.bc['dhfzi'] = np.zeros_like(layer.bc['dh'])
    layer.bc['dhfz'] = np.zeros_like(layer.bc['dh'])
    layer.bc['dhn'] = np.zeros(layer.fs['h'])

    layer.bc['dA'] = np.zeros(layer.fs['X'])

    dhn = np.zeros_like(layer.bc['dh'][:, 0])

    return dX, dhn


def gru_backward(layer, dA):
    """Backward propagate signal through GRU cells to previous layer.
    """
    dX, dhn = initialize_backward(layer, dA)

    # Loop through steps
    for s in reversed(range(layer.d['s'])):

        dX = layer.bc['dX'][:, s] if layer.sequences else dX

        dh =  dX + dhn

        dhh = dh * (1-layer.fc['z'][:, s])
        dhh = layer.bc['dhh'][:, s] = dhh * layer.activate(layer.fc['hh'][:, s], deriv=True)

        dr = np.dot(dhh, layer.p['Wh'].T)
        dr = dr * layer.fc['h'][:, s - 1]
        dr = layer.bc['dr'][:, s] = dr * layer.activate_reset(layer.fc['r'][:, s], deriv=True)

        dz = dh * (layer.fc['h'][:, s - 1] - layer.fc['hh'][:, s])
        dz = layer.bc['dz'][:, s] = dz * layer.activate_update(layer.fc['z'][:, s], deriv=True)

        dhfhh = np.dot(dhh, layer.p['Wh'].T)
        dhfhh = layer.bc['dhfhh'][:, s] = dhfhh * layer.fc['r'][:, s]
        dhfr = layer.bc['dhfr'][:, s] = np.dot(dr, layer.p['Wr'].T)
        dhfzi = layer.bc['dhfzi'][:, s] = np.dot(dz, layer.p['Wz'].T)
        dhfz = layer.bc['dhfz'][:, s] = dh * layer.fc['z'][:, s]
        dhn = layer.bc['dhn'][:, s] = dhfhh + dhfr + dhfzi + dhfz

        dA = np.dot(dr, layer.p['Ur'].T)
        dA += np.dot(dz, layer.p['Uz'].T)
        dA += np.dot(dhh, layer.p['Uh'].T)
        layer.bc['dA'][:, s] = dA

    dA = layer.bc['dA']

    return dA    # To previous layer
