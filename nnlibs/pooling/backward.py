# EpyNN/nnlibs/pool/backward.py
# Related third party imports
import numpy as np


def initialize_backward(layer, dX):
    """Backward cache initialization.

    :param layer: An instance of pooling layer.
    :type layer: :class:`nnlibs.pooling.models.Pooling`

    :param dX: Output of backward propagation from next layer.
    :type dX: :class:`numpy.ndarray`

    :return: Input of backward propagation for current layer.
    :rtype: :class:`numpy.ndarray`
    """
    dA = layer.bc['dA'] = dX

    return dA


def pooling_backward(layer, dX):
    """Backward propagate error gradients to previous layer.
    """
    # (1) Initialize cache
    dA = initialize_backward(layer, dX)

    # dA.shape = (m, oh, ow, d)
    dA = np.repeat(dA, layer.d['zh'], axis=1)      # (m, zh * oh, ow, d)
    dA = np.repeat(dA, layer.d['zw'], axis=2)      # (m, zh * oh, zw * ow, d)
    dA = np.pad(dA, layer.bs['p'], mode='constant', constant_values=0)

    # Z.shape = (m, oh, ow, d)
    mask = layer.fc['Z']
    mask = np.repeat(mask, layer.d['zh'], axis=1)  # (m, zh * oh, ow, d)
    mask = np.repeat(mask, layer.d['zw'], axis=2)  # (m, zh * oh, zw * ow, d)
    mask = np.pad(mask, layer.bs['p'], mode='constant', constant_values=0)

    #
    mask = (layer.fc['X'] == mask)

    dA = dA * mask

    dX = np.zeros_like(layer.fc['X'])

    for h in range(layer.d['oh']):

        hs = h * layer.d['sh']
        he = hs + layer.d['ph']

        for w in range(layer.d['ow']):

            ws = w * layer.d['sw']
            we = ws + layer.d['pw']

            dXb = dA[:, hs:he, ws:we, :]

            dX[:, hs:he, ws:we, :] += dXb

    layer.bc['dX'] = dX

    return dX
