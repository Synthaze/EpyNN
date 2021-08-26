# EpyNN/nnlibs/convolution/backward.py
# Related third party imports
import numpy as np


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

    dZ = np.expand_dims(dZ, axis=3)
    dZ = np.expand_dims(dZ, axis=3)
    dZ = np.expand_dims(dZ, axis=3)

    dX = np.zeros_like(layer.fc['X'])

    for h in range(dA.shape[1]):

        hs = h * layer.d['sh']
        he = hs + layer.d['fh']

        for w in range(dA.shape[2]):

            ws = w * layer.d['sw']
            we = ws + layer.d['fw']

            dXb = dZ[:, h, w, :] * layer.p['W']

            dX[:, hs:he, ws:we, :] += np.sum(dXb, axis=4)

    return dX
