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

    layer.bc['dX'] = np.zeros(layer.fs['X'])

    return dA


def convolution_backward(layer, dX):
    """Backward propagate error to previous layer.
    """
    # (1) Initialize cache
    dA = initialize_backward(layer, dX)

    dZ = layer.bc['dZ'] = dA * layer.activate(layer.fc['Z'], deriv=True)

    block = np.expand_dims(dZ, axis=3)
    block = np.expand_dims(block, axis=3)
    block = np.expand_dims(block, axis=3)

    # () Gradients of the loss with respect to X
    dX = block * layer.p['W']             # ()
    dX = np.sum(dX, axis=6)               # ()
    dX = np.reshape(dX, layer.fs['X'])    # ()

    # Remove padding
    dX = layer.bc['dX'] = padding(dX, layer.d['p'], forward=False)

    return dX
