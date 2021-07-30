# EpyNN/nnlibs/dense/forward.py
# Related third party imports
import numpy as np


def initialize_forward(layer, A):
    """

    :param layer: An instance of the :class:`nnlibs.dense.models.Dense`
    :type layer: class:`nnlibs.dense.models.Dense`

    :param A: Output of forward propagation from previous layer
    :type A: class:`numpy.ndarray`

    :return: Input of forward propagation for current layer
    :rtype: class:`numpy.ndarray`
    """
    X = layer.fc['X'] = A

    return X


def dense_forward(layer, A):
    """

    :param layer: An instance of the :class:`nnlibs.dense.models.Dense`
    :type layer: class:`nnlibs.dense.models.Dense`

    :param A: Output of forward propagation from previous layer
    :type A: class:`numpy.ndarray`

    :return: Output of forward propagation for current layer
    :rtype: class:`numpy.ndarray`
    """

    X = initialize_forward(layer, A)

    Z = layer.fc['Z'] = np.dot(layer.p['W'], X) + layer.p['b']

    A = layer.fc['A'] = layer.activate(Z)

    return A    # To next layer
