# EpyNN/nnlibs/pool/backward.py
# Related third party imports
import numpy as np


def initialize_backward(layer, dA):
    """Backward cache initialization.
    :param layer: An instance of pooling layer.
    :type layer: :class:`nnlibs.pooling.models.Pooling`
    :param dA: Output of backward propagation from next layer.
    :type dA: :class:`numpy.ndarray`
    :return: Input of backward propagation for current layer.
    :rtype: :class:`numpy.ndarray`
    :return: Zeros-output of backward propagation for current layer.
    :rtype: :class:`numpy.ndarray`
    """
    dX = layer.bc['dX'] = dA

    layer.fc['dA'] = np.zeros(layer.fs['X'])

    return dX


def pooling_backward(layer, dA):
    """Backward propagate error to previous layer.
    """
    # (1) Initialize cache
    dX = initialize_backward(layer, dA)

    # Iterate over image rows
    for h in range(layer.d['oh']):

        #
        hs = h * layer.d['sh']
        hsp = (layer.d['ih'] - hs) % layer.d['ph']

        #
        mask_row = layer.fc['Z'][:, hs::layer.d['oh'], :, :]

        #
        row = dX[:, hs::layer.d['oh'], :, :]

        # Iterate over image columns
        for w in range(layer.d['ow']):

            #
            ws = w * layer.d['sw']
            wsp = (layer.d['iw'] - ws) % layer.d['pw']

            #
            mask = mask_row[:, :, ws::layer.d['ow'], :]
            mask = assemble_block(layer, mask, h, hsp, w, wsp)

            #
            block = row[:, :, ws::layer.d['ow'], :]
            block = assemble_block(layer, block, h, hsp, w, wsp)

            #
            mask = (layer.fc['X'][:, h:layer.d['ih'] - hsp, w:layer.d['iw'] - wsp, :] == mask)

            #
            layer.fc['dA'][:, h:layer.d['ih'] - hsp, w:layer.d['iw'] - wsp, :] += block * mask

    dA = layer.fc['dA']

    return dA


def assemble_block(layer, block, h, hsp, w, wsp):
    """.
    """
    block = np.repeat(block, layer.d['ph'], axis=1)
    block = np.repeat(block, layer.d['pw'], axis=2)
    block = np.array(np.split(block, block.shape[2] / layer.d['ph'], axis=2))
    block = np.array(np.split(block, block.shape[2] / layer.d['pw'], axis=2))
    block = np.moveaxis(block, 0, 2)
    block = np.moveaxis(block, 0, 3)
    block = np.reshape(block, (layer.d['m'], layer.d['ih'] - h - hsp, layer.d['iw'] - w - wsp, layer.d['n']))

    return block
