# EpyNN/nnlibs/dense/backward.py
# Related third party imports
import numpy as np


def initialize_backward(layer, dA):
    """Backward cache initialization.

    :param layer: An instance of dense layer.
    :type layer: :class:`nnlibs.dense.models.Dense`

    :param dA: Output of backward propagation from next layer.
    :type dA: :class:`numpy.ndarray`

    :return: Input of backward propagation for current layer.
    :rtype: :class:`numpy.ndarray`
    """
    dX = layer.bc['dX'] = dA

    return dX


def dense_backward(layer, dA):
    """Backward propagate error to previous layer.
    """
    # (1) Initialize cache
    dX = initialize_backward(layer, dA)

    # (2) Gradients of the loss with respect to Z
    dZ = layer.bc['dZ'] = dX * layer.activate(layer.fc['Z'], deriv=True)

    # (3)
    dA = layer.bc['dA'] = np.dot(dZ, layer.p['W'].T)

    return dA    # To previous layer
