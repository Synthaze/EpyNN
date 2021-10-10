# EpyNN/epynn/pooling/forward.py
# Related third party imports
import numpy as np


def initialize_forward(layer, A):
    """Forward cache initialization.

    :param layer: An instance of pooling layer.
    :type layer: :class:`epynn.pooling.models.Pooling`

    :param A: Output of forward propagation from previous layer.
    :type A: :class:`numpy.ndarray`

    :return: Input of forward propagation for current layer.
    :rtype: :class:`numpy.ndarray`

    :return: Input of forward propagation for current layer.
    :rtype: :class:`numpy.ndarray`

    :return: Input blocks of forward propagation for current layer.
    :rtype: :class:`numpy.ndarray`
    """
    X = layer.fc['X'] = A

    return X


def pooling_forward(layer, A):
    """Forward propagate signal to next layer.
    """
    # (1) Initialize cache
    X = initialize_forward(layer, A)

    # (2) Slice input w.r.t. pool size (ph, pw) and strides (sh, sw)
    Xb = np.array([[X[ :, h:h + layer.d['ph'], w:w + layer.d['pw'], :]
                    # Inner loop
                    # (m, h, w, d) ->
                    # (ow, m, h, pw, d)
                    for w in range(layer.d['w'] - layer.d['pw'] + 1)
                    if w % layer.d['sw'] == 0]
                # Outer loop
                # (ow, m, h, pw, d) ->
                # (oh, ow, m, ph, pw, d)
                for h in range(layer.d['h'] - layer.d['ph'] + 1)
                if h % layer.d['sh'] == 0])

    # (3) Bring back m along axis 0
    Xb = layer.fc['Xb'] = np.moveaxis(Xb, 2, 0)
    # (oh, ow, m, ph, pw, d) ->
    # (m, oh, ow, ph, pw, d)

    # (4) Apply pooling operation on blocks
    Z = layer.pool(Xb, axis=(4, 3))
    # (m, oh, ow, ph, pw, d) - Xb
    # (m, oh, ow, ph, d)     - layer.pool(Xb, axis=4)
    # (m, oh, ow, d)         - layer.pool(Xb, axis=(4, 3))

    A = layer.fc['A'] = layer.fc['Z'] = Z

    return A    # To next layer
