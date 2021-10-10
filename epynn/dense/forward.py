# EpyNN/epynn/dense/forward.py
# Related third party imports
import numpy as np


def initialize_forward(layer, A):
    """Forward cache initialization.

    :param layer: An instance of dense layer.
    :type layer: :class:`epynn.dense.models.Dense`

    :param A: Output of forward propagation from previous layer.
    :type A: :class:`numpy.ndarray`

    :return: Input of forward propagation for current layer.
    :rtype: :class:`numpy.ndarray`
    """
    X = layer.fc['X'] = A

    return X


def dense_forward(layer, A):
    """Forward propagate signal to next layer.
    """
    # (1) Initialize cache
    X = initialize_forward(layer, A)

    # (2) Linear activation X -> Z
    Z = layer.fc['Z'] = (
        np.dot(X, layer.p['W'])
        + layer.p['b']
    )   # This is the weighted sum

    # (3) Non-linear activation Z -> A
    A = layer.fc['A'] = layer.activate(Z)

    return A    # To next layer
