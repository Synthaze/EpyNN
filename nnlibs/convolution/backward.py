# EpyNN/nnlibs/conv/backward.py
# Related third party imports
from nnlibs.commons.io import padding
import numpy as np


def initialize_backward(layer, dA):
    """Backward cache initialization.

    :param layer: An instance of convolution layer.
    :type layer: :class:`nnlibs.convolution.models.Convolution`

    :param dA: Output of backward propagation from next layer.
    :type dA: :class:`numpy.ndarray`

    :return: Input of backward propagation for current layer.
    :rtype: :class:`numpy.ndarray`

    :return: Zeros-output of backward propagation for current layer.
    :rtype: :class:`numpy.ndarray`
    """
    dX = layer.bc['dX'] = dA

    layer.bc['dA'] = np.zeros(layer.fs['X'])

    return dX


def convolution_backward(layer, dA):
    """Backward propagate error to previous layer.
    """
    # (1) Initialize cache
    dX = initialize_backward(layer, dA)

    # Iterate over image rows
    for t in range(layer.d['oh']):

        #
        row = layer.bc['dX'][:, t::layer.d['oh'], :, :]

        # Iterate over image columns
        for l in range(layer.d['ow']):

            #
            b = (layer.d['ih'] - t * layer.d['s']) % layer.d['w']
            r = (layer.d['iw'] - l * layer.d['s']) % layer.d['w']

            # () Extract block
            block = row[:, :, l * layer.d['s']::layer.d['ow'], :]

            #
            block = np.expand_dims(block, axis=3)
            block = np.expand_dims(block, axis=3)
            block = np.expand_dims(block, axis=3)

            #
            dA = block * layer.p['W']

            #
            dA = np.sum(dA, axis=6)

            #
            dA = np.reshape(dA, (
                                layer.d['m'],
                                layer.d['ih'] - b - t,
                                layer.d['iw'] - r - l,
                                layer.d['id']
                                )
                            )
            layer.bc['dA'][:, t:layer.d['ih'] - b, l:layer.d['iw'] - r, :] += dA

    # Remove padding
    dA = layer.bc['dA'] = padding(layer.bc['dA'], layer.d['p'], forward=False)

    return dA    # To previous layer
