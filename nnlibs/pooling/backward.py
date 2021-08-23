# EpyNN/nnlibs/pool/backward.py
# Related third party imports
import numpy as np


def initialize_backward(layer, dA):
    """Backward cache initialization.

    :param layer: An instance of pooling layer.
    :type layer: :class:`nnlibs.pooling.models.Pooling`

    :param dA: Output of backward propagation from next layer.
    :type dA: :class:`numpy.ndarray`

    :return: Input of backward propagation for current layer.
    :rtype: :class:`numpy.ndarray`

    :return: Zeros-output of backward propagation for current layer.
    :rtype: :class:`numpy.ndarray`
    """
    dX = layer.bc['dX'] = dA

    layer.fc['dA'] = np.zeros(layer.fs['X'])

    return dX


def pooling_backward(layer, dA):
    """Backward propagate error to previous layer.
    """
    # (1) Initialize cache
    dX = initialize_backward(layer, dA)

    mask = np.repeat(layer.fc['Z'], layer.d['ph'], axis=1)
    mask = np.repeat(mask, layer.d['pw'], axis=2)

    block = np.repeat(dX, layer.d['ph'], axis=1)
    block = np.repeat(block, layer.d['pw'], axis=2)

    mask = (layer.fc['X'] == mask)

    dA = layer.fc['dA'] = block * mask

    return dA
