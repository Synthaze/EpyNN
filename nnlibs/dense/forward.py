# EpyNN/nnlibs/dense/forward.py
# Related third party imports
import numpy as np


def initialize_forward(layer, A):
    """Forward cache initialization.

    :param layer: An instance of dense layer.
    :type layer: :class:`nnlibs.dense.models.Dense`

    :param A: Output of forward propagation from previous layer
    :type A: :class:`numpy.ndarray`

    :return: Input of forward propagation for current layer
    :rtype: :class:`numpy.ndarray`
    """
    X = layer.fc['X'] = A

    return X


def dense_forward(layer, A):
    """Forward propagate signal to next layer.
    """
    # (1) Initialize cache
    X = initialize_forward(layer, A)

    # (2)
    Z = layer.fc['Z'] = np.dot(X, layer.p['W']) + layer.p['b']

    # (3)
    A = layer.fc['A'] = layer.activate(Z)

    return A    # To next layer
