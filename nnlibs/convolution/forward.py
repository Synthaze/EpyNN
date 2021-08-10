# EpyNN/nnlibs/convolution/forward.py
# Related third party imports
from nnlibs.commons.io import padding
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
    X = layer.fc['X'] = padding(A, layer.d['p'])

    Z = np.empty(layer.fs['Z'])

    layer.Xb = []

    return X, Z


def convolution_forward(layer, A):
    """Forward propagate signal to next layer.
    """
    # (1) Initialize cache
    X, Z = initialize_forward(layer, A)

    # layer.p['W'] = None
    #
    # if layer.p['W'] is None:
    #     layer.p['W'] = layer.initialization(layer.fs['W'], rng=layer.np_rng)
    #     layer.p['b'] = np.zeros(layer.fs['b'])

    for t in range(layer.d['oh']):

        layer.Xb.append([])

        b = layer.d['ih'] - (layer.d['ih'] - t) % layer.d['w']

        cols_shape = (
                     layer.d['m'],
                     int((b - t) / layer.d['w']),
                     layer.d['zw'],
                     layer.d['n']
                     )

        cols = np.empty(cols_shape)


        for i in range(layer.d['ow']):

            l = i * layer.d['s']
            r = layer.d['iw'] - (layer.d['iw'] - l) % layer.d['w']

            block = X[:, t:b, l:r, :]

            block = np.array(np.split(block, (r - l) / layer.d['w'], 2))
            block = np.array(np.split(block, (b - t) / layer.d['w'], 2))

            block = np.moveaxis(block, 2, 0)
            block = np.expand_dims(block, axis=6)

            layer.Xb[t].append(block)

            block = block * layer.p['W']

            block = np.sum(block, axis=5)
            block = np.sum(block, axis=4)
            block = np.sum(block, axis=3)

            cols[:, :, i::layer.d['ow'], :] = block

        Z[:, t * layer.d['s'] ::layer.d['oh'], :, :] = cols

#    Z += layer.p['b']

    layer.fc['Z'] = Z

    A = layer.fc['A'] = layer.activate(Z)

    return A
