# EpyNN/epynn/dropout/backward.py
# Related third party imports
import numpy as np


def initialize_backward(layer, dX):
    """Backward cache initialization.

    :param layer: An instance of dropout layer.
    :type layer: :class:`epynn.dropout.models.Dropout`

    :param dX: Output of backward propagation from next layer.
    :type dX: :class:`numpy.ndarray`

    :return: Input of backward propagation for current layer.
    :rtype: :class:`numpy.ndarray`
    """
    dA = layer.bc['dA'] = dX

    return dA


def dropout_backward(layer, dX):
    """Backward propagate error gradients to previous layer.
    """
    # (1) Initialize cache
    dA = initialize_backward(layer, dX)

    # (2) Apply the dropout mask used in the forward pass
    dX = dA * layer.fc['D']

    # (3) Scale up gradients
    dX /= (1 - layer.d['d'])

    layer.bc['dX'] = dX

    return dX    # To previous layer
