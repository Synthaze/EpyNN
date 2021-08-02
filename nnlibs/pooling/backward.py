# EpyNN/nnlibs/pool/backward.py
# Related third party imports
import numpy as np


def initialize_backward(layer, dA):
    """Backward cache initialization.

    :param layer: An instance of RNN layer.
    :type layer: :class:`nnlibs.rnn.models.RNN`

    :param dA: Output of backward propagation from next layer
    :type dA: :class:`numpy.ndarray`

    :return: Input of backward propagation for current layer
    :rtype: :class:`numpy.ndarray`

    :return: Zeros-output of backward propagation for current layer
    :rtype: :class:`numpy.ndarray`
    """
    # Cache X (current) from A (prev)
    dX = layer.bc['dX'] = dA

    dA = np.zeros(layer.fs['X'])

    return dX, dA


def pooling_backward(layer, dA):
    """Backward propagate signal to previous layer.
    """
    # (1) Initialize cache
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
                X = layer.fc['X'][m, ih1:ih2, iw1:iw2, n]
                dA[m, ih1:ih2, iw1:iw2, n] += np.sum(dX[m , h, w, n] * (X == np.max(X)))

    layer.bc['dA'] = dA

    return dA
