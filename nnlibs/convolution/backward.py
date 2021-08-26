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
    """Backward propagate error gradients to previous layer.
    """
    # (1) Initialize cache
    dA = initialize_backward(layer, dX)    # (m, oh, ow, u)

    # (2) Gradient of the loss with respect to Z
    dZ = layer.bc['dZ'] = dA * layer.activate(layer.fc['Z'], deriv=True)

    # (3) Restore filter units kernel dimensions
    dZ = np.expand_dims(dZ, axis=3)    # (m, oh, ow, d, u)
    dZ = np.expand_dims(dZ, axis=3)    # (m, oh, ow, fw, d, u)
    dZ = np.expand_dims(dZ, axis=3)    # (m, oh, ow, fh, fw, d, u)

    # (4)
    dX = np.zeros_like(layer.fc['X'])  # (m, h, w, d)

    for h in range(layer.d['oh']):

        hs = h * layer.d['sh']
        he = hs + layer.d['fh']

        for w in range(layer.d['ow']):

            ws = w * layer.d['sw']
            we = ws + layer.d['fw']

            dXb = dZ[:, h, w, :] * layer.p['W']

            dX[:, hs:he, ws:we, :] += np.sum(dXb, axis=4)

    layer.bc['dX'] = dX

    return dX
