# EpyNN/epynn/gru/forward.py
# Related third party imports
import numpy as np


def initialize_forward(layer, A):
    """Forward cache initialization.

    :param layer: An instance of GRU layer.
    :type layer: :class:`epynn.gru.models.GRU`

    :param A: Output of forward propagation from previous layer.
    :type A: :class:`numpy.ndarray`

    :return: Input of forward propagation for current layer.
    :rtype: :class:`numpy.ndarray`

    :return: Previous hidden state initialized with zeros.
    :rtype: :class:`numpy.ndarray`
    """
    X = layer.fc['X'] = A

    cache_keys = ['h', 'hp', 'hh_', 'hh', 'z', 'z_', 'r', 'r_']
    layer.fc.update({k: np.zeros(layer.fs['h']) for k in cache_keys})

    h = layer.fc['h'][:, 0]    # Hidden state

    return X, h


def gru_forward(layer, A):
    """Forward propagate signal to next layer.
    """
    # (1) Initialize cache and hidden state
    X, h = initialize_forward(layer, A)

    # Iterate over sequence steps
    for s in range(layer.d['s']):

        # (2s) Slice sequence (m, s, e) with respect to step
        X = layer.fc['X'][:, s]

        # (3s) Retrieve previous hidden state
        hp = layer.fc['hp'][:, s] = h

        # (4s) Activate reset gate
        r_ = layer.fc['r_'][:, s] = (
            np.dot(X, layer.p['Ur'])
            + np.dot(hp, layer.p['Vr'])
            + layer.p['br']
        )   # (4.1s)

        r = layer.fc['r'][:, s] = layer.activate_reset(r_)    # (4.2s)

        # (5s) Activate update gate
        z_ = layer.fc['z_'][:, s] = (
            np.dot(X, layer.p['Uz'])
            + np.dot(hp, layer.p['Vz'])
            + layer.p['bz']
        )   # (5.1s)

        z = layer.fc['z'][:, s] = layer.activate_update(z_)   # (5.2s)

        # (6s) Activate hidden hat
        hh_ = layer.fc['hh_'][:, s] = (
            np.dot(X, layer.p['Uhh'])
            + np.dot(r * hp, layer.p['Vhh'])
            + layer.p['bhh']
        )   # (6.1s)

        hh = layer.fc['hh'][:, s] = layer.activate(hh_)       # (6.2s)

        # (7s) Compute current hidden state
        h = layer.fc['h'][:, s] = (
            z*hp
            + (1-z)*hh
        )

    # Return the last hidden state or the full sequence
    A = layer.fc['h'] if layer.sequences else layer.fc['h'][:, -1]

    return A   # To next layer
