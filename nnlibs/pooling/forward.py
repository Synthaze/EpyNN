# EpyNN/nnlibs/pooling/forward.py
# Related third party imports
import numpy as np

# Local application/library specific imports
from nnlibs.commons.io import extract_blocks


def initialize_forward(layer, A):
    """Forward cache initialization.

    :param layer: An instance of pooling layer.
    :type layer: :class:`nnlibs.pooling.models.Pooling`

    :param A: Output of forward propagation from previous layer.
    :type A: :class:`numpy.ndarray`

    :return: Input of forward propagation for current layer.
    :rtype: :class:`numpy.ndarray`

    :return: Input of forward propagation for current layer.
    :rtype: :class:`numpy.ndarray`

    :return: Input blocks of forward propagation for current layer.
    :rtype: :class:`numpy.ndarray`
    """
    X = layer.fc['X'] = A

    pool_size = (layer.d['ph'], layer.d['pw'])
    strides = (layer.d['sh'], layer.d['sw'])

    Xb = extract_blocks(X, pool_size, strides)

    return X, Xb


def pooling_forward(layer, A):
    """Forward propagate signal to next layer.
    """
    # (1) Initialize cache and extract X blocks
    X, Xb = initialize_forward(layer, A)

    # (2) Block pooling with respect to features depth (d)
    Xb = layer.pool(Xb, axis=(4, 3))    # (width, height)

    Z = Xb

    A = layer.fc['A'] = layer.fc['Z'] = Z

    return A    # To next layer
