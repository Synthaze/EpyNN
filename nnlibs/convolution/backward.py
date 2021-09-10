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

    # (2) Gradient of the loss w.r.t. Z
    dZ = layer.bc['dZ'] = (
        dA
        * layer.activate(layer.fc['Z'], deriv=True)
    )   # dL/dZ

    # (3) Restore kernel dimensions
    dZb = dZ
    dZb = np.expand_dims(dZb, axis=3)
    dZb = np.expand_dims(dZb, axis=3)
    dZb = np.expand_dims(dZb, axis=3)
    # (m, oh, ow, d, u) ->
    # (m, oh, ow, fw, d, u) ->
    # (m, oh, ow, fh, fw, d, u)

    # (4) Initialize backward output dL/dX
    dX = np.zeros_like(layer.fc['X'])      # (m, h, w, d)

    # Iterate over forward output height
    for h in range(layer.d['oh']):

        hs = h * layer.d['sh']
        he = hs + layer.d['fh']

        # Iterate over forward output width
        for w in range(layer.d['ow']):

            ws = w * layer.d['sw']
            we = ws + layer.d['fw']

            # (5hw) Gradient of the loss w.r.t Xb
            dXb = dZb[:, h, w, :] * layer.p['W']
            # (m, oh, ow, fh, fw, d, u) - dZb
            #         (m, fh, fw, d, u) - dZb[:, h, w, :]
            #            (fh, fw, d, u) - W

            # (6hw) Sum over units axis
            dX[:, hs:he, ws:we, :] += np.sum(dXb, axis=4)
            # (m, fh, fw, d, u) - dXb
            # (m, fh, fw, d)    - np.sum(dXb, axis=4)

    layer.bc['dX'] = dX

    return dX
