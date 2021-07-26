# EpyNN/nnlibs/pool/backward.py
# Related third party imports
import numpy as np


def pooling_backward(layer, dA):

    dX, dA = initialize_backward(layer, dA)

    # Loop through image rows
    for t in range(layer.d['R']):

        mask_row = layer.fc['Z'][:, t::layer.d['R'], :, :]

        row = dX[:, t::layer.d['R'], :, :]

        # Loop through row columns
        for l in range(layer.d['C']):

            # region of X and dZ for this block
            b = (layer.d['ih'] - t * layer.d['s']) % layer.d['w']
            # region of X and dZ for this block
            r = (layer.d['iw'] - l * layer.d['s']) % layer.d['w']
            # _
            mask = mask_row[:, :, l * layer.d['s']::layer.d['C'], :]
            mask = layer.assemble_block(mask, t, b, l, r)
            # _
            block = row[:, :, l * layer.d['s']::layer.d['C'], :]
            block = layer.assemble_block(block, t, b, l, r)
            # _
            mask = (layer.fc['X'][:, t:layer.d['ih'] - b, l:layer.d['iw'] - r, :] == mask)
            # _
            dA[:, t:layer.d['ih'] - b, l:layer.d['iw'] - r, :] += block * mask

    dA = layer.bc['dA'] = dA

    return dA


def initialize_backward(layer, dA):

    # Cache X (current) from A (prev)
    dX = layer.bc['dX'] = dA

    dA = np.zeros(layer.fs['X'])

    return dX, dA
