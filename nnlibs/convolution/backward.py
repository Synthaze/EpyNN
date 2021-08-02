# EpyNN/nnlibs/conv/backward.py
# Related third party imports
import numpy as np


def initialize_backward(layer, dA):
    """Backward cache initialization.

    :param layer: An instance of convolution layer.
    :type layer: :class:`nnlibs.convolution.models.Convolution`

    :param dA: Output of backward propagation from next layer
    :type dA: :class:`numpy.ndarray`

    :return: Input of backward propagation for current layer
    :rtype: :class:`numpy.ndarray`

    :return: Zeros-output of backward propagation for current layer
    :rtype: :class:`numpy.ndarray`
    """
    dX = layer.bc['dX'] = dA

    dA = np.zeros(layer.fs['X'])

    return dX, dA


def convolution_backward(layer, dA):
    """Backward propagate signal to previous layer.
    """
    dX, dA = initialize_backward(layer, dA)

    #
    for m in range(layer.d['m']):
        #
        for h in range(layer.d['oh']):
            ih1 = h * layer.d['s']
            ih2 = ih1 + layer.d['w']
            #
            for w in range(layer.d['ow']):
                iw1 = w * layer.d['s']
                iw2 = iw1 + layer.d['w']
            #
            for n in range(layer.d['n']):
                dA[m, ih1:ih2, iw1:iw2, :] +=  layer.p['W'][:, :, :, n] * dX[m, h, w, n]

    layer.bc['dA'] = dA

    return dA




#
#
#
# def restore_padding(layer,dA):
#
#     if layer.d['p'] > 0:
#         p = layer.d['p']
#         dA = dA[:, p:-p, p:-p, :]
#
#     return dA
