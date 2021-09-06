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
    dA = initialize_backward(layer, dX)    # (m, oh, ow, d)

    # (2) dA shape (m, oh, ow, d) to X shape (m, h, w, d)
    dAx = dA
    dAx = np.repeat(dAx, layer.d['zh'], axis=1)
    dAx = np.repeat(dAx, layer.d['zw'], axis=2)
    dAx = np.pad(dAx, layer.bs['p'], mode='constant', constant_values=0)
    # (m, oh, ow, d)                       -> np.repeat(dAx, zh, axis=1)
    # (m, zh * oh, ow, d)                  -> np.repeat(dAx, zw, axis=2)
    # (m, zh * oh, zw * ow, d)             -> padding
    # (m, zh * oh + p1, zw * ow + p2, d)   <->
    # (m, h, w, d)

    # (3) Z shape (m, oh, ow, d) to X shape (m, h, w, d)
    Zx = layer.fc['Z']
    Zx = np.repeat(Zx, layer.d['zh'], axis=1)
    Zx = np.repeat(Zx, layer.d['zw'], axis=2)
    Zx = np.pad(Zx, layer.bs['p'], mode='constant', constant_values=0)
    # (m, oh, ow, d)                       -> np.repeat(Zx, zh, axis=1)
    # (m, zh * oh, ow, d)                  -> np.repeat(Zx, zw, axis=2)
    # (m, zh * oh, zw * ow, d)             -> padding
    # (m, zh * oh + p1, zw * ow + p2, d)   <->
    # (m, h, w, d)

    # (4) Map coordinates in X corresponding to the pooled values in Z
    mask = (layer.fc['X'] == Zx)
    # e.g.
    # X    = [[[[0], [2], [3], [1]]]] with shape = (1, 1, 4, 1) <-> (m, h, w, d)
    # Z    = [[[[3],]]]               with shape = (1, 1, 1, 1) <-> (m, oh, ow, d)
    # Zx   = [[[[3], [3], [3], [3]]]] with shape = (1, 1, 4, 1) <-> (m, h, w, d)
    # mask = [[[[0], [0], [1], [0]]]] with shape = (1, 1, 4, 1) <-> (m, h, w, d)

    # (5) Retain or set values to zero in dL/dA w.r.t mask
    mdAx = dAx * mask
    # dA    = [[[[-1],]]]                  with shape = (1, 1, 1, 1) <-> (m, oh, ow, d)
    # dAx   = [[[[-1], [-1], [-1], [-1]]]] with shape = (1, 1, 4, 1) <-> (m, h, w, d)
    # mask  = [[[[0], [0], [1], [0]]]]     with shape = (1, 1, 4, 1) <-> (m, h, w, d)
    # mdAx  = [[[[0], [0], [-1], [0]]]]    with shape = (1, 1, 4, 1) <-> (m, h, w, d)

    # (6) Initialize backward output dL/dX
    dX = np.zeros_like(layer.fc['X'])
    # (m, h, w, d)

    # Iterate over forward output height
    for h in range(layer.d['oh']):

        hs = h * layer.d['sh']
        he = hs + layer.d['ph']

        # Iterate over forward output width
        for w in range(layer.d['ow']):

            ws = w * layer.d['sw']
            we = ws + layer.d['pw']

            # (7) Slice block from masked dA having same shape as X
            dXb = mdAx[:, hs:he, ws:we, :]

            # (8) Increment dL/dX with dL/dXb at consistent coordinates
            dX[:, hs:he, ws:we, :] += dXb

    layer.bc['dX'] = dX

    return dX
