# EpyNN/epynn/convolution/forward.py
# Related third party imports
import numpy as np

# Local application/library specific imports
from epynn.commons.io import padding


def initialize_forward(layer, A):
    """Forward cache initialization.

    :param layer: An instance of convolution layer.
    :type layer: :class:`epynn.convolution.models.Convolution`

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
    X = initialize_forward(layer, A)    # (m, h, w, d)

    # (2) Slice input w.r.t. filter size (fh, fw) and strides (sh, sw)
    Xb = np.array([[X[ :, h:h + layer.d['fh'], w:w + layer.d['fw'], :]
                    # Inner loop
                    # (m, h, w, d) ->
                    # (ow, m, h, fw, d)
                    for w in range(layer.d['w'] - layer.d['fw'] + 1)
                    if w % layer.d['sw'] == 0]
                # Outer loop
                # (ow, m, h, fw, d) ->
                # (oh, ow, m, fh, fw, d)
                for h in range(layer.d['h'] - layer.d['fh'] + 1)
                if h % layer.d['sh'] == 0])

    # (3) Bring back m along axis 0
    Xb = np.moveaxis(Xb, 2, 0)
    # (oh, ow, m, fh, fw, d) ->
    # (m, oh, ow, fh, fw, d)

    # (4) Add dimension for filter units (u) on axis 6
    Xb = layer.fc['Xb'] = np.expand_dims(Xb, axis=6)
    # (m, oh, ow, fh, fw, d) ->
    # (m, oh, ow, fh, fw, d, 1)

    # (5.1) Linear activation Xb -> Zb
    Zb = Xb * layer.p['W']
    # (m, oh, ow, fh, fw, d, 1) - Xb
    #            (fh, fw, d, u) - W

    # (5.2) Sum block products
    Z = np.sum(Zb, axis=(5, 4, 3))
    # (m, oh, ow, fh, fw, d, u) - Zb
    # (m, oh, ow, fh, fw, u)    - np.sum(Zb, axis=(5))
    # (m, oh, mw, fh, u)        - np.sum(Zb, axis=(5, 4))
    # (m, oh, ow, u)            - np.sum(Zb, axis=(5, 4, 3))

    # (5.3) Add bias to linear activation product
    Z = layer.fc['Z'] = Z + layer.p['b']

    # (6) Non-linear activation
    A = layer.fc['A'] = layer.activate(Z)

    return A    # To next layer
