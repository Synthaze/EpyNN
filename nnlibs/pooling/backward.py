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

    layer.fc['dA'] = np.zeros(layer.fs['X'])

    return dX


def pooling_backward(layer, dA):
    """Backward propagate signal to previous layer.
    """
    # (1) Initialize cache
    dX = initialize_backward(layer, dA)

    dA_prev = np.zeros(layer.fs['X'])

    for t in range(layer.d['oh']):

        mask_row = layer.fc['Z'][:, t::layer.d['oh'], :, :]
        row = dX[:, t::layer.d['oh'], :, :]

        for l in range(layer.d['ow']):
            
            b = (layer.d['ih'] - t * layer.d['s']) % layer.d['w']
            r = (layer.d['iw'] - l * layer.d['s']) % layer.d['w']

            mask = mask_row[:, :, l * layer.d['s']::layer.d['ow'], :]

            mask = assemble_block(layer, mask, t, b, l, r)

            block = row[:, :, l * layer.d['s']::layer.d['ow'], :]

            block = assemble_block(layer, block, t, b, l, r)

            mask = (layer.fc['X'][:, t:layer.d['ih'] - b, l:layer.d['iw'] - r, :] == mask)

            layer.fc['dA'][:, t:layer.d['ih'] - b, l:layer.d['iw'] - r, :] += block * mask

    dA = layer.fc['dA']

    return dA



def assemble_block(layer, block, t, b, l, r):
    block = np.repeat(block, layer.d['w'] ** 2, 2)
    block = np.array(np.split(block, block.shape[2] / layer.d['w'], 2))
    block = np.moveaxis(block, 0, 2)
    block = np.array(np.split(block, block.shape[2] / layer.d['w'], 2))
    block = np.moveaxis(block, 0, 3)
    block = np.reshape(block, (layer.d['m'], layer.d['ih'] - t - b, layer.d['iw'] - l - r, layer.d['id']))
    return block
