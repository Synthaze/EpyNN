# EpyNN/epynn/rnn/forward.py
# Related third party imports
import numpy as np


def initialize_forward(layer, A):
    """Forward cache initialization.

    :param layer: An instance of RNN layer.
    :type layer: :class:`epynn.rnn.models.RNN`

    :param A: Output of forward propagation from previous layer.
    :type A: :class:`numpy.ndarray`

    :return: Input of forward propagation for current layer.
    :rtype: :class:`numpy.ndarray`

    :return: Previous hidden state initialized with zeros.
    :rtype: :class:`numpy.ndarray`
    """
    X = layer.fc['X'] = A

    cache_keys = ['h_', 'h', 'hp']
    layer.fc.update({k: np.zeros(layer.fs['h']) for k in cache_keys})

    h = layer.fc['h'][:, 0]    # Hidden state

    return X, h


def rnn_forward(layer, A):
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

        # (4s) Activate current hidden state
        h_ = layer.fc['h_'][:, s] = (
            np.dot(X, layer.p['U'])
            + np.dot(hp, layer.p['V'])
            + layer.p['b']
        )   # (4.1s) Linear

        h = layer.fc['h'][:, s] = layer.activate(h_)   # (4.2s) Non-linear

    # Return the last hidden state or the full sequence
    A = layer.fc['h'] if layer.sequences else layer.fc['h'][:, -1]

    return A   # To next layer
