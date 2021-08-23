# EpyNN/nnlibs/rnn/forward.py
# Related third party imports
import numpy as np


def initialize_forward(layer, A):
    """Forward cache initialization.

    :param layer: An instance of RNN layer.
    :type layer: :class:`nnlibs.rnn.models.RNN`

    :param A: Output of forward propagation from previous layer.
    :type A: :class:`numpy.ndarray`

    :return: Input of forward propagation for current layer.
    :rtype: :class:`numpy.ndarray`

    :return: Previous cell state initialized with zeros.
    :rtype: :class:`numpy.ndarray`
    """
    X = layer.fc['X'] = A

    cache_keys = ['h', 'hp']

    layer.fc.update({k: np.zeros(layer.fs['h']) for k in cache_keys})

    h = layer.fc['h'][:, 0]

    return X, h


def rnn_forward(layer, A):
    """Forward propagate signal through RNN cells to next layer.
    """
    # (1) Initialize cache and cell state
    X, h = initialize_forward(layer, A)

    # Iterate over sequence steps
    for s in range(layer.d['s']):

        # (2s) Slice sequence (m, s, v) with respect to step
        X = layer.fc['X'][:, s]

        #
        hp = layer.fc['hp'][:, s] = h

        # (3s) Activate hidden cell state
        h = np.dot(X, layer.p['U'])
        h += np.dot(hp, layer.p['W'])
        h += layer.p['b']

        h = layer.fc['h'][:, s] = layer.activate(h)

    # Return all or only the last hidden cell state
    A = layer.fc['h'] if layer.sequences else layer.fc['h'][:, -1]

    layer.fc['A'] = A

    return A   # To next layer
