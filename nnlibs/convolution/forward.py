# EpyNN/nnlibs/convolution/forward.py
# Related third party imports
import numpy as np

# Local application/library specific imports
from nnlibs.commons.io import padding, extract_blocks


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

    return X


def convolution_forward(layer, A):
    """Forward propagate signal to next layer.
    """
    # (1) Initialize cache and pad image
    X = initialize_forward(layer, A)

    #
    Xb = np.array([[X[ :, h:h + layer.d['fh'], w:w + layer.d['fw'], :]
                    for w in range(layer.d['w'] - layer.d['fw'] + 1)
                    if w % layer.d['sw'] == 0]
                    for h in range(layer.d['h'] - layer.d['fh'] + 1)
                    if h % layer.d['sh'] == 0])

    #
    Xb = np.moveaxis(Xb, 2, 0)

    # (m, mh, mw, fh, fw, d) -> (m, mh, mw, fh, fw, d, u)
    Xb = layer.fc['Xb'] = np.expand_dims(Xb, axis=6)

    # (2) Linear activation Xb -> Zb
    Zb = Xb * layer.p['W']

    # (3) Sum block products
    Z = Zb                   # (m, oh, ow, fh, fw, d, u)
    Z = np.sum(Z, axis=5)    # (m, oh, ow, fh, fw, u)
    Z = np.sum(Z, axis=4)    # (m, oh, mw, fh, u)
    Z = np.sum(Z, axis=3)    # (m, oh, ow, u)

    # (4) Add bias to linear activation product
    Z = layer.fc['Z'] = Z + layer.p['b']

    # (5) Non-linear activation
    A = layer.fc['A'] = layer.activate(Z)

    return A    # To next layer
