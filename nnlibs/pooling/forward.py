# EpyNN/nnlibs/pooling/forward.py
# Related third party imports
import numpy as np


def initialize_forward(layer, A):
    """Forward cache initialization.

    :param layer: An instance of pooling layer.
    :type layer: :class:`nnlibs.pooling.models.Pooling`

    :param A: Output of forward propagation from previous layer.
    :type A: :class:`numpy.ndarray`

    :return: Input of forward propagation for current layer.
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

    # Iterate over image rows
    for h in range(layer.d['oh']):

        #
        hs = h * layer.d['sh']
        hp = layer.d['ih'] - (layer.d['ih'] - h) % layer.d['ph']

        layer.fs['Zw'] = (
                     layer.d['m'],
                     int((hp - h) / layer.d['ph']),
                     layer.d['zw'],
                     layer.d['n']
                     )

        #
        Zw = np.empty(layer.fs['Zw'])

        # Iterate over image columns
        for w in range(layer.d['ow']):

            #
            ws = w * layer.d['sw']
            wsp = layer.d['iw'] - (layer.d['iw'] - ws) % layer.d['pw']

            # () Extract block of shape
            block = X[:, h:hp, ws:wsp]

            #
            block = np.array(np.split(block, (wsp - ws) / layer.d['pw'], axis=2))
            block = np.array(np.split(block, (hp - h) / layer.d['ph'], axis=2))

            #
            block = layer.pool(block, axis=4)
            block = layer.pool(block, axis=3)

            #
            block = np.moveaxis(block, 0, 2)
            block = np.moveaxis(block, 0, 2)

            #
            Zw[:, :, w::layer.d['ow'], :] = block

        #
        Z[:, hs::layer.d['oh'],:] = Zw

    A = layer.fc['A'] = layer.fc['Z'] = Z

    return A    # To next layer
