# EpyNN/nnlibs/dropout/backward.py
# Related third party imports
import numpy as np


def initialize_backward(layer, dA):
    """Backward cache initialization.

    :param layer: An instance of dropout layer.
    :type layer: :class:`nnlibs.dropout.models.Dropout`

    :param dA: Output of backward propagation from next layer
    :type dA: :class:`numpy.ndarray`

    :return: Input of backward propagation for current layer
    :rtype: :class:`numpy.ndarray`
    """
    dX = layer.bc['dX'] = dA

    return dX


def dropout_backward(layer, dA):
    """Backward propagate signal to previous layer.
    """
    # (1)
    dX = initialize_backward(layer, dA)

    # (2)
    dA = dX * layer.fc['D']
    dA /= layer.d['k']

    dA = layer.bc['dA'] = dA

    return dA    # To previous layer
