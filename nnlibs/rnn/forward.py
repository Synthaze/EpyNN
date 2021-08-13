# EpyNN/nnlibs/rnn/forward.py
# Related third party imports
import numpy as np


def initialize_forward(layer, A):
    """Forward cache initialization.

    :param layer: An instance of RNN layer.
    :type layer: :class:`nnlibs.rnn.models.RNN`

    :param A: Output of forward propagation from previous layer
    :type A: :class:`numpy.ndarray`

    :return: Input of forward propagation for current layer
    :rtype: :class:`numpy.ndarray`

    :return: Previous cell state initialized with zeros.
    :rtype: :class:`numpy.ndarray`
    """
    X = layer.fc['X'] = A

    layer.fc['h'] = np.zeros(layer.fs['h'])

    hp = np.zeros_like(layer.fc['h'][:, 0])

    return X, hp


def rnn_forward(layer, A):
    """Forward propagate signal through RNN cells to next layer.
    """
    # (1) Initialize cache and cell state
    X, hp = initialize_forward(layer, A)

    # Iterate through sequence to next cell
    for s in range(layer.d['s']):

        # (2s) Slice sequence (l, v, m) with respect to step
        X = layer.fc['X'][:, s]

        # (3s) Compute cell state
        h = np.dot(X, layer.p['U'])
        h += np.dot(hp, layer.p['W'])
        h += layer.p['b']

        h = hp = layer.fc['h'][:, s] = layer.activate(h)

    #
    A = layer.fc['h'] if layer.sequences else layer.fc['h'][:, -1]

    layer.fc['A'] = A

    return A   # To next layer
