# EpyNN/nnlibs/rnn/forward.py
# Related third party imports
import numpy as np


def initialize_forward(layer, A):
    """Forward cache initialization.

    :param layer: RNN layer object
    :type layer: class:`nnlibs.rnn.models.RNN`
    :param A: Output of forward propagation from previous layer
    :type A: class:`numpy.ndarray`

    :return: Input of forward propagation for current layer
    :rtype: class:`numpy.ndarray`
    :return: Previous cell state initialized with zeros.
    :rtype: class:`numpy.ndarray`
    """
    X = layer.fc['X'] = np.zeros(layer.fs['X'])
    X[:layer.d['s']] = A

    layer.fc['h'] = np.zeros(layer.fs['h'])
    layer.fc['A'] = np.zeros(layer.fs['A'])

    hp = np.zeros_like(layer.fc['h'][0])

    return X, hp


def rnn_forward(layer, A):
    """Forward propagate signal to next layer.
    """
    # (1) Initialize cache and cell state
    X, hp = initialize_forward(layer, A)

    # Iterate through sequence to next cell
    for s in range(layer.d['h']):

        # (2s) Slice sequence (l, v, m) with respect to step
        X = layer.fc['X'][s]

        # (3s) Compute cell state
        h = np.dot(layer.p['Wx'], X)
        h += np.dot(layer.p['Wh'], hp) + layer.p['bh']
        h = hp = layer.fc['h'][s] = layer.activate_hidden(h)

        # (4s) Compute cell output to next layer
        A = np.dot(layer.p['W'], h) + layer.p['b']
        A = layer.fc['A'][s] = layer.activate(A)

    # Return layer.fc['A'] if many-to-many mode
    A = terminate_forward(layer)

    return A    # To next layer


def terminate_forward(layer):
    """Check if many-to-many or many-to-one (binary) mode.

    :param layer: RNN layer object
    :type layer: class:`nnlibs.rnn.models.RNN`

    :return: Output of forward propagation for current layer
    :rtype: class:`numpy.ndarray`
    """
    layer.fc['A'] = layer.fc['A'][-1] if layer.binary else layer.fc['A']

    A = layer.fc['A']

    return A
