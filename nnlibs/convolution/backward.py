# EpyNN/nnlibs/conv/backward.py
# Related third party imports
from nnlibs.commons.io import padding
import numpy as np


def initialize_backward(layer, dX):
    """Backward cache initialization.

    :param layer: An instance of convolution layer.
    :type layer: :class:`nnlibs.convolution.models.Convolution`

    :param dX: Output of backward propagation from next layer.
    :type dX: :class:`numpy.ndarray`

    :return: Input of backward propagation for current layer.
    :rtype: :class:`numpy.ndarray`
    """
    dA = layer.bc['dA'] = dX

    layer.bc['dX'] = np.zeros(layer.fs['X'])

    return dA


def convolution_backward(layer, dX):
    """Backward propagate error to previous layer.
    """
    # (1) Initialize cache
    dA = initialize_backward(layer, dX)

    dZ = layer.bc['dZ'] = dA * layer.activate(layer.fc['Z'], deriv=True)

    # Iterate over image rows
    for h in range(layer.d['oh']):

        #
        hs = h * layer.d['sh']
        hsf = (layer.d['ih'] - hs) % layer.d['fh']

        #
        row = layer.bc['dZ'][:, h::layer.d['oh'], :, :]

        # Iterate over image columns
        for w in range(layer.d['ow']):

            #
            ws = w * layer.d['sw']
            wsf = (layer.d['iw'] - ws) % layer.d['fw']

            # () Extract block
            block = row[:, :, ws::layer.d['ow'], :]

            #
            block = np.expand_dims(block, axis=3)
            block = np.expand_dims(block, axis=3)
            block = np.expand_dims(block, axis=3)

            # () Gradients of the loss with respect to X
            dX = block * layer.p['W']    # ()
            dX = np.sum(dX, axis=6)      # ()
            dX = np.reshape(dX, (
                                layer.d['m'],
                                layer.d['ih'] - hsf - h,
                                layer.d['iw'] - wsf - w,
                                layer.d['id']
                                )
                            )            # ()

            layer.bc['dX'][:, h:layer.d['ih'] - hsf, w:layer.d['iw'] - wsf, :] += dX

    # Remove padding
    dX = layer.bc['dX'] = padding(layer.bc['dX'], layer.d['p'], forward=False)

    return dX    # To previous layer
