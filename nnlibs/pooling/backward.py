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

    # (3) Restore pooling block axes
    dZb = dA
    dZb = np.expand_dims(dZb, axis=3)
    dZb = np.expand_dims(dZb, axis=3)
    # (m, oh, ow, d)         ->
    # (m, oh, ow, 1, d)      ->
    # (m, oh, ow, 1, 1, d)

    # (4) Initialize backward output dL/dX
    dX = np.zeros_like(layer.fc['X'])      # (m, h, w, d)

    # Iterate over forward output height
    for h in range(layer.d['oh']):

        hs = h * layer.d['sh']
        he = hs + layer.d['ph']

        # Iterate over forward output width
        for w in range(layer.d['ow']):

            ws = w * layer.d['sw']
            we = ws + layer.d['pw']

            # (5hw) Retrieve input block
            Xb = layer.fc['Xb'][:, h, w, :, :, :]
            # (m, oh, ow, ph, pw, d)  - Xb (array of blocks)
            # (m, ph, pw, d)          - Xb (single block)

            # (6hw) Retrieve pooled value and restore block shape
            Zb = layer.fc['Z'][:, h:h+1, w:w+1, :]
            Zb = np.repeat(Zb, layer.d['ph'], axis=1)
            Zb = np.repeat(Zb, layer.d['pw'], axis=2)
            # (m, oh, ow, d)    - Z
            # (m,  1,  1, d)    - Zb -> np.repeat(Zb, pw, axis=1)
            # (m, ph,  1, d)         -> np.repeat(Zb, pw, axis=2)
            # (m, ph, pw, d)

            # (7hw) Match pooled value in Zb against Xb
            mask = (Zb == Xb)

            # (8hw) Retrieve gradient w.r.t pooled Xb and restore block shape
            dXb = dZb[:, h, w, :]
            dXb = np.repeat(dXb, layer.d['ph'], 1)
            dXb = np.repeat(dXb, layer.d['pw'], 2)
            # (m, oh, ow,  1,  1, d) - dZb
            #         (m,  1,  1, d) - dXb -> np.repeat(dXb, ph, axis=1)
            #         (m, ph,  1, d)       -> np.repeat(dXb, pw, axis=2)
            #         (m, ph, pw, d)

            # (9hw) Keep dXb values for coordinates where Zb = Xb (mask)
            dXb = dXb * mask

            # (10hw) Gradient of the loss w.r.t Xb
            dX[:, hs:he, ws:we, :] += dXb
            # (m, ph, pw, d) - dX[:, hs:he, ws:we, :]
            # (m, ph, pw, d) - dXb

    layer.bc['dX'] = dX

    return dX
