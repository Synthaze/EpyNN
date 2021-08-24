# EpyNN/nnlibs/convolution/forward.py
# Related third party imports
import numpy as np

# Local application/library specific imports
from nnlibs.commons.io import (
    padding,
    extract_blocks,
)


def initialize_forward(layer, A):
    """Forward cache initialization.

    :param layer: An instance of convolution layer.
    :type layer: :class:`nnlibs.convolution.models.Convolution`

    :param A: Output of forward propagation from previous layer.
    :type A: :class:`numpy.ndarray`

    :return: Input of forward propagation for current layer.
    :rtype: :class:`numpy.ndarray`

    :return: Input of forward propagation for current layer.
    :rtype: :class:`numpy.ndarray`

    :return: Input blocks of forward propagation for current layer.
    :rtype: :class:`numpy.ndarray`
    """
    X = layer.fc['X'] = padding(A, layer.d['p'])

    sizes = (layer.d['fh'], layer.d['fw'])
    strides = (layer.d['sh'], layer.d['sw'])

    Xb = extract_blocks(X, sizes, strides)

    return X, Xb


def convolution_forward(layer, A):
    """Forward propagate signal to next layer.
    """
    # (1) Initialize cache and pad image
    X, Xb = initialize_forward(layer, A)

    # (m, mh, mw, fh, fw, d) -> (m, mh, mw, fh, fw, d, u)
    Xb = layer.fc['Xb'] = np.expand_dims(Xb, axis=6)

    # (2) Linear activation Xb -> Zb
    Zb = Xb * layer.p['W']

    # (3) Sum block products
    Z = Zb                           # (m, mh, mw, fh, fw, d, u)
    Z = np.sum(Z, axis=(5, 4, 3))    # (m, mh, mw, u)

    # (4) Add bias to linear activation product
    Z = layer.fc['Z'] = Z  if layer.use_bias else Z

    # (5) Non-linear activation
    A = layer.fc['A'] = layer.activate(Z)

    return A    # To next layer
