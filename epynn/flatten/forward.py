# EpyNN/epynn/flatten/forward.py
# Related third party imports
import numpy as np


def initialize_forward(layer, A):
    """Forward cache initialization.

    :param layer: An instance of flatten layer.
    :type layer: :class:`epynn.flatten.models.Flatten`

    :param A: Output of forward propagation from previous layer.
    :type A: :class:`numpy.ndarray`

    :return: Input of forward propagation for current layer.
    :rtype: :class:`numpy.ndarray`
    """
    X = layer.fc['X'] = A

    return X


def flatten_forward(layer,A):
    """Forward propagate signal to next layer.
    """
    # (1) Initialize cache
    X = initialize_forward(layer, A)

    # (2) Reshape  (m, ...) -> (m, n)
    A = layer.fc['A'] = np.reshape(X, layer.fs['A'])

    return A    # To next layer
