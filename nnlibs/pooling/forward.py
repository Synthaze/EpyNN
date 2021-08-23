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

    :return:
    :rtype: :class:`numpy.ndarray`
    """
    X = layer.fc['X'] = A

    sizes = (layer.d['ph'], layer.d['pw'])
    strides = (layer.d['sh'], layer.d['sw'])

    Xb = layer.fc['Xb'] = extract_blocks(X, sizes, strides)

    return X, Xb


def pooling_forward(layer, A):
    """Forward propagate signal to next layer.
    """
    # (1) Initialize cache
    X, Xb = initialize_forward(layer, A)

    #
    Xb = layer.pool(Xb, axis=4)
    Xb = layer.pool(Xb, axis=3)

    #
    Xb = np.moveaxis(Xb, 0, 2)
    Xb = np.moveaxis(Xb, 0, 2)

    Z = Xb

    A = layer.fc['A'] = layer.fc['Z'] = Z

    return A    # To next layer
