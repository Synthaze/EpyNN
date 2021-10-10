# EpyNN/epynn/dense/backward.py
# Related third party imports
import numpy as np

# Local application/library specific imports
from epynn.commons.maths import hadamard


def initialize_backward(layer, dX):
    """Backward cache initialization.

    :param layer: An instance of dense layer.
    :type layer: :class:`epynn.dense.models.Dense`

    :param dX: Output of backward propagation from next layer.
    :type dX: :class:`numpy.ndarray`

    :return: Input of backward propagation for current layer.
    :rtype: :class:`numpy.ndarray`
    """
    dA = layer.bc['dA'] = dX

    return dA


def dense_backward(layer, dX):
    """Backward propagate error gradients to previous layer.
    """
    # (1) Initialize cache
    dA = initialize_backward(layer, dX)

    # (2) Gradient of the loss with respect to Z
    dZ = layer.bc['dZ'] = hadamard(
         dA,
         layer.activate(layer.fc['Z'], deriv=True)
    )    # dL/dZ

    # (3) Gradient of the loss with respect to X
    dX = layer.bc['dX'] = np.dot(dZ, layer.p['W'].T)   # dL/dX

    return dX    # To previous layer
