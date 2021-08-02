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

    Z = np.empty(layer.fs['Z'])

    return X, Z


def pooling_forward(layer, A):
    """Forward propagate signal to next layer.
    """
    # (1) Initialize cache
    X, Z = initialize_forward(layer, A)

    # Loop through image rows
    for t in range(layer.d['R']):

        b = layer.d['ih'] - (layer.d['ih'] - t) % layer.d['w']

        ax1 = int((b - t) / layer.d['w'])

        z_shape = (layer.d['im'], ax1, layer.d['zw'], layer.d['id'])

        Z_cols = np.empty(z_shape)

        # Loop through row columns
        for i in range(layer.d['C']):

            # _
            l = i * layer.d['s']
            r = layer.d['iw'] - (layer.d['iw'] - l) % layer.d['w']

            # _
            Xb = X.T[:, t:b, l:r, :]

            # _
            Xb = np.array(np.split(Xb, (r - l) / layer.d['w'], 2))
            Xb = np.array(np.split(Xb, (b - t) / layer.d['w'], 2))

            # _
            Xb = layer.pool(Xb, 4)
            Xb = layer.pool(Xb, 3)

            # _
            Xb = np.moveaxis(Xb, 0, 2)
            Xb = np.moveaxis(Xb, 0, 2)

            # _
            Z_cols[:, :, i::layer.d['C'], :] = Xb

        # _
        Z[:, t * layer.d['s'] ::layer.d['R'], :, :] = Z_cols

    A = layer.fc['Z'] = layer.fc['A'] = Z

    return A    # To next layer
