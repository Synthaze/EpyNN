# EpyNN/nnlibs/convolution/backward.py
# Related third party imports
import numpy as np

# Local application/library specific imports
from nnlibs.commons.io import padding


def initialize_backward(layer, dX):
    """Backward cache initialization.

    :param layer: An instance of convolution layer.
    :type layer: :class:`nnlibs.convolution.models.Convolution`

    :param dX: Output of backward propagation from next layer.
    :type dX: :class:`numpy.ndarray`

    :return: Input of backward propagation for current layer.
    :rtype: :class:`numpy.ndarray`
    """
    dA = layer.bc['dA'] = dX

    return dA


def convolution_backward(layer, dX):
    """Backward propagate error to previous layer.
    """
    # (1) Initialize cache
    dA = initialize_backward(layer, dX)

    # (2) Gradient of the loss with respect to Z
    dZ = layer.bc['dZ'] = dA * layer.activate(layer.fc['Z'], deriv=True)

    # (3) Expand dZ (m, mh, mw, u)
    dZ = np.expand_dims(dZ, axis=3)    # (m, mh, mw, fh, u)
    dZ = np.expand_dims(dZ, axis=3)    # (m, mh, mw, fh, fw, u)
    dZ = np.expand_dims(dZ, axis=3)    # (m, mh, mw, fh, fw, d, u)

    # (4) Gradients of the loss with respect to X
    dX = dZ * layer.p['W']                # (4.1) X blocks
    dX = np.sum(dX, axis=6)               # (4.2) X
    dX = np.reshape(dX, layer.fs['X'])    # (4.3) Reshape dX

    # Remove zeros-padding of feature planes (h, w)
    dX = layer.bc['dX'] = padding(dX, layer.d['p'], forward=False)

    return dX
