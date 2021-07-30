# EpyNN/nnlibs/flatten/forward.py
# Related third party imports
import numpy as np


def initialize_forward(layer, A):
    """

    :param layer: An instance of the :class:`nnlibs.flatten.models.Flatten`
    :type layer: class:`nnlibs.flatten.models.Flatten`

    :param A: Output of forward propagation from previous layer
    :type A: class:`numpy.ndarray`

    :return: Input of forward propagation for current layer
    :rtype: class:`numpy.ndarray`
    """
    X = layer.fc['X'] = A

    return X


def flatten_forward(layer,A):
    """

    :param layer: An instance of the :class:`nnlibs.flatten.models.Flatten`
    :type layer: class:`nnlibs.flatten.models.Flatten`

    :param A: Output of forward propagation from previous layer
    :type A: class:`numpy.ndarray`

    :return: Output of forward propagation for current layer
    :rtype: class:`numpy.ndarray`
    """
    X = initialize_forward(layer, A)

    A = layer.fc['A'] = np.reshape(X, layer.fs['A'])

    return A   # To next layer
