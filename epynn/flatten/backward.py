# EpyNN/epynn/flatten/backward.py
# Related third party imports
import numpy as np


def initialize_backward(layer, dX):
    """Backward cache initialization.

    :param layer: An instance of flatten layer.
    :type layer: :class:`epynn.flatten.models.Flatten`

    :param dX: Output of backward propagation from next layer.
    :type dX: :class:`numpy.ndarray`

    :return: Input of backward propagation for current layer.
    :rtype: :class:`numpy.ndarray`
    """
    dA = layer.bc['dA'] = dX

    return dA


def flatten_backward(layer, dX):
    """Backward propagate error gradients to previous layer.
    """
    # (1)
    dA = initialize_backward(layer, dX)

    # (2) Reverse reshape (m, n) -> (m, ...)
    dX = layer.bc['dX'] = np.reshape(dA, layer.fs['X'])

    return dX    # To previous layer
