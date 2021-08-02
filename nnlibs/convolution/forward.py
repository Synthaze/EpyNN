# EpyNN/nnlibs/convolution/forward.py
# Related third party imports
import numpy as np


def initialize_forward(layer, A):
    """Forward cache initialization.

    :param layer: An instance of convolution layer.
    :type layer: :class:`nnlibs.convolution.models.Convolution`

    :param A: Output of forward propagation from previous layer
    :type A: :class:`numpy.ndarray`

    :return: Input of forward propagation for current layer
    :rtype: :class:`numpy.ndarray`
    """
    X = layer.fc['X'] = A

    Z = np.empty(layer.fs['Z'])

    layer.fc['Xb'] = []

    return X, Z


def convolution_forward(layer, A):
    """Forward propagate signal to next layer.
    """
    # (1) Initialize cache
    X, Z = initialize_forward(layer, A)

    # Loop through image rows
    for t in range(layer.d['R']):

        layer.fc['Xb'].append([])

        b = layer.d['ih'] - (layer.d['ih'] - t) % layer.d['w']

        ax1 = int((b - t) / layer.d['w'])

        c_shape = (layer.d['im'], ax1, layer.d['zh'], layer.d['n'])

        cols = np.empty(c_shape)

        # Loop through row columns
        for i in range(layer.d['C']):

            # _
            l = i * layer.d['s']

            r = layer.d['iw'] - (layer.d['iw'] - l) % layer.d['w']

            # _
            Xb = X[:, t:b, l:r, :]

            # _
            Xb = np.array(np.split(Xb, (r - l) / layer.d['w'], 2))
            Xb = np.array(np.split(Xb, (b - t) / layer.d['w'], 2))

            # _
            Xb = np.moveaxis(Xb, 2, 0)

            # _
            Xb = np.expand_dims(Xb, 6)

            # _
            layer.fc['Xb'][t].append(Xb)

            # _
            Xb *= layer.p['W']

            # _
            Xb = np.sum(Xb, axis=(5, 4, 3))

            # _
            cols[:, :, i::layer.d['C'], :] = Xb

        # _
        Z[:, t * layer.d['s'] ::layer.d['R'], :, :] = cols

    # _
    Z = layer.fc['Z'] = Z + layer.p['b']

    # _
    A = layer.fc['A'] = layer.activate(Z)

    return A    # To next layer
