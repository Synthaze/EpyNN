# EpyNN/nnlibs/flatten/backward.py
# Related third party imports
import numpy as np


def initialize_backward(layer, dA):
    """Backward cache initialization.

    :param layer: An instance of flatten layer.
    :type layer: :class:`nnlibs.flatten.models.Flatten`

    :param dA: Output of backward propagation from next layer.
    :type dA: :class:`numpy.ndarray`

    :return: Input of backward propagation for current layer.
    :rtype: :class:`numpy.ndarray`
    """
    dX = layer.bc['dX'] = dA

    return dX


def flatten_backward(layer, dA):
    """Backward propagate error to previous layer.
    """
    # (1)
    dX = initialize_backward(layer, dA)

    # (2) Reshape (m, sv) to (m, s, v)
    dA = layer.bc['dA'] = np.reshape(dX, layer.fs['X'])

    return dA    # To previous layer
