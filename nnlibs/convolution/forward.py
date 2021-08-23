# EpyNN/nnlibs/convolution/forward.py
# Related third party imports
from nnlibs.commons.io import padding
import numpy as np


def initialize_forward(layer, A):
    """Forward cache initialization.

    :param layer: An instance of convolution layer.
    :type layer: :class:`nnlibs.convolution.models.Convolution`

    :param A: Output of forward propagation from previous layer.
    :type A: :class:`numpy.ndarray`

    :return: Input of forward propagation for current layer.
    :rtype: :class:`numpy.ndarray`
    """
    X = layer.fc['X'] = padding(A, layer.d['p'])

    Z = np.empty(layer.fs['Z'])

    layer.Xb = []

    return X, Z


def convolution_forward(layer, A):
    """Forward propagate signal to next layer.
    """
    # (1) Initialize cache and pad image
    X, Z = initialize_forward(layer, A)

    # Iterate over image rows
    for h in range(layer.d['oh']):

        layer.Xb.append([])

        hs = h * layer.d['sh']
        hf = layer.d['ih'] - (layer.d['ih'] - h) % layer.d['fh']

        layer.fs['Zw'] = (
                     layer.d['m'],
                     int((hf - h) / layer.d['fh']),
                     layer.d['zw'],
                     layer.d['n']
                     )

        Zw = np.empty(layer.fs['Zw'])

        # Iterate over image columns
        for w in range(layer.d['ow']):

            #
            ws = w * layer.d['sw']
            wsf = layer.d['iw'] - (layer.d['iw'] - ws) % layer.d['fw']

            # () Extract block of shape (m, b - t, r - l, id)
            block = X[:, h:hf, ws:wsf, :]

            #
            block = np.array(np.split(block, (wsf - ws) / layer.d['fw'], axis=2))
            block = np.array(np.split(block, (hf - h) / layer.d['fh'], axis=2))

            #
            block = np.moveaxis(block, 2, 0)

            #
            block = np.expand_dims(block, axis=6)

            layer.Xb[w].append(block)

            # () Linear activation
            block = block * layer.p['W']

            #
            block = np.sum(block, axis=5)
            block = np.sum(block, axis=4)
            block = np.sum(block, axis=3)

            Zw[:, :, w::layer.d['ow'], :] = block

        #
        Z[:, hs::layer.d['oh'], :, :] = Zw

    # () Add bias to linear activation product
    Z = layer.fc['Z'] = Z + layer.p['b'] if layer.use_bias else Z

    # () Non-linear activation
    A = layer.fc['A'] = layer.activate(Z)

    return A    # To next layer
