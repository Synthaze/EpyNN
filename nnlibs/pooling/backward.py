# EpyNN/nnlibs/pool/backward.py
# Related third party imports
import numpy as np


def initialize_backward(layer, dX):
    """Backward cache initialization.

    :param layer: An instance of pooling layer.
    :type layer: :class:`nnlibs.pooling.models.Pooling`

    :param dX: Output of backward propagation from next layer.
    :type dX: :class:`numpy.ndarray`

    :return: Input of backward propagation for current layer.
    :rtype: :class:`numpy.ndarray`
    """
    dA = layer.bc['dA'] = dX

    return dA


def pooling_backward(layer, dX):
    """Backward propagate error to previous layer.
    """
    # (1) Initialize cache
    dA = initialize_backward(layer, dX)

    # (2) Recover dimensions for error
    dA = np.repeat(dA, layer.d['ph'], axis=1)
    dA = np.repeat(dA, layer.d['pw'], axis=2)

    # (3) Recover dimension for mask
    mask = layer.fc['Z']
    mask = np.repeat(mask, layer.d['ph'], axis=1)
    mask = np.repeat(mask, layer.d['pw'], axis=2)

    # (4) Mapping pooling output to input
    mask = (layer.fc['X'] == mask)

    # (5) Preserve gradients
    dX = layer.fc['dX'] = dA * mask

    return dX
