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

    :return: Zeros-output of backward propagation for current layer.
    :rtype: :class:`numpy.ndarray`
    """
    dA = layer.bc['dA'] = dX

    layer.fc['dX'] = np.zeros(layer.fs['X'])

    return dA


def pooling_backward(layer, dX):
    """Backward propagate error to previous layer.
    """
    # (1) Initialize cache
    dA = initialize_backward(layer, dX)

    mask = np.repeat(layer.fc['Z'], layer.d['ph'], axis=1)
    mask = np.repeat(mask, layer.d['pw'], axis=2)

    block = np.repeat(dA, layer.d['ph'], axis=1)
    block = np.repeat(block, layer.d['pw'], axis=2)

    mask = (layer.fc['X'] == mask)

    dX = layer.fc['dX'] = block * mask

    return dX
