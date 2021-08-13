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

    cache_keys = ['h', 'hh', 'z', 'r']

    layer.fc.update({k: np.zeros(layer.fs['h']) for k in cache_keys})

    hp = np.zeros_like(layer.fc['h'][:, 0])

    return X, hp


def gru_forward(layer, A):
    """Forward propagate signal through GRU cells to next layer.
    """
    # (1) Initialize cache and cell state
    X, hp = initialize_forward(layer, A)

    # Loop through steps
    for s in range(layer.d['s']):

        # (2s)
        X = layer.fc['X'][:, s]

        # (3s)
        r = np.dot(X, layer.p['Ur'])
        r += np.dot(hp, layer.p['Wr'])
        r += layer.p['br']

        r = layer.fc['r'][:, s] = layer.activate_reset(r)

        # (4s)
        z = np.dot(X, layer.p['Uz'])
        z += np.dot(hp, layer.p['Wz'])
        z += layer.p['bz']

        z = layer.fc['z'][:, s] = layer.activate_update(z)

        # (5s)
        hh = np.dot(X, layer.p['Uh'])
        hh += np.dot(r * hp, layer.p['Wh'])
        hh += layer.p['bh']

        hh = layer.fc['hh'][:, s] = layer.activate(hh)

        # (6s)
        h = hp = layer.fc['h'][:, s] = z*hp + (1-z)*hh

    #
    A = layer.fc['h'] if layer.sequences else layer.fc['h'][:, -1]

    layer.fc['A'] = A

    return A   # To next layer
