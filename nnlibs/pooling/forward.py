# EpyNN/nnlibs/pooling/forward.py
# Related third party imports
import numpy as np


def initialize_forward(layer, A):
    """Forward cache initialization.

    :param layer: An instance of pooling layer.
    :type layer: :class:`nnlibs.pooling.models.Pooling`

    :param A: Output of forward propagation from previous layer
    :type A: :class:`numpy.ndarray`

    :return: Input of forward propagation for current layer
    :rtype: :class:`numpy.ndarray`

    :return:
    :rtype: :class:`numpy.ndarray`
    """
    X = layer.fc['X'] = A

    Z = layer.fc['Z'] = np.empty(layer.fs['Z'])

    return X, Z


def pooling_forward(layer, A):
    """Forward propagate signal to next layer.
    """
    # (1) Initialize cache
    X, Z = initialize_forward(layer, A)


    for t in range(layer.d['oh']):

        b = layer.d['ih'] - (layer.d['ih'] - t) % layer.d['w']

        Z_cols_shape = (
                       layer.d['m'],
                       int((b - t) / layer.d['w']),
                       layer.d['zw'],
                       layer.d['id']
                       )

        Z_cols = np.empty(Z_cols_shape)


        for i in range(layer.d['ow']):

            l = i * layer.d['s']
            r = layer.d['iw'] - (layer.d['iw'] - l) % layer.d['w']

            block = X[:, t:b, l:r, :]

            block = np.array(np.split(block, (r - l) / layer.d['w'], 2))
            block = np.array(np.split(block, (b - t) / layer.d['w'], 2))

            block = layer.pool(block, 4)
            block = layer.pool(block, 3)

            block = np.moveaxis(block, 0, 2)
            block = np.moveaxis(block, 0, 2)

            Z_cols[:, :, i::layer.d['ow'], :] = block

        Z[:, t * layer.d['s']::layer.d['oh'], :, :] = Z_cols

    layer.fc['Z'] = Z

    return Z
