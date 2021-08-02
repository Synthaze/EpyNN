# EpyNN/nnlibs/gru/forward.py
# Related third party imports
import numpy as np


def initialize_forward(layer, A):
    """Forward cache initialization.

    :param layer: An instance of GRU layer.
    :type layer: :class:`nnlibs.gru.models.GRU`

    :param A: Output of forward propagation from previous layer
    :type A: :class:`numpy.ndarray`

    :return: Input of forward propagation for current layer
    :rtype: :class:`numpy.ndarray`

    :return: Previous cell state initialized with zeros.
    :rtype: :class:`numpy.ndarray`
    """
    X = layer.fc['X'] = A

    layer.fc['h'] = np.zeros(layer.fs['h'])
    layer.fc['hh'] = np.zeros_like(layer.fc['h'])
    layer.fc['z'] = np.zeros_like(layer.fc['h'])
    layer.fc['r'] = np.zeros_like(layer.fc['h'])

    hp = np.zeros_like(layer.fc['h'][:, 0])

    return X, hp


def gru_forward(layer, A):
    """Forward propagate signal through GRU cells to next layer.
    """
    # (1) Initialize cache and cell state
    X, hp = initialize_forward(layer, A)

    # Loop through steps
    for s in range(layer.d['s']):

        X = layer.fc['X'][:, s]

        r = np.dot(X, layer.p['Ur'])
        r += np.dot(hp, layer.p['Wr'])
        r += layer.p['br']
        r = layer.fc['r'][:, s] = layer.activate_reset(r)

        z = np.dot(X, layer.p['Uz'])
        z += np.dot(hp, layer.p['Wz'])
        z += layer.p['bz']
        z = layer.fc['z'][:, s] = layer.activate_update(z)

        hh = np.dot(X, layer.p['Uh'])
        hh += np.dot(r * hp, layer.p['Wh'])
        hh += layer.p['bh']
        hh = layer.fc['hh'][:, s] = layer.activate(hh)

        h = hp = layer.fc['h'][:, s] = z*hp + (1-z)*hh

    A = layer.fc['A'] = layer.fc['h']

    return A   # To next layer
