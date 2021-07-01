#EpyNN/nnlibs/pool/backward.py
import numpy as np


def pooling_backward(layer,dA):

    # Cache X (current) from A (prev)
    dX = layer.bc['dX'] = dA
    im, ih, iw, id = layer.fs['X']

    dA = np.zeros( layer.fs['X'] )

    # Loop through image rows
    for t in range(layer.n_rows):

        mask_row = layer.fc['Z'][:, t::layer.n_rows, :, :]

        row = dX[:, t::layer.n_rows, :, :]

        # Loop through row columns
        for l in range(layer.n_cols):

            # region of X and dZ for this block
            b = (ih - t * layer.d['s']) % layer.d['fw']
            # region of X and dZ for this block
            r = (iw - l * layer.d['s']) % layer.d['fw']
            # _
            mask = mask_row[:, :, l * layer.d['s']::layer.n_cols, :]
            mask = layer.assemble_block(mask, t, b, l, r)
            # _
            block = row[:, :, l * layer.d['s']::layer.n_cols, :]
            block = layer.assemble_block(block, t, b, l, r)
            # _
            mask = (layer.fc['X'][:, t:ih - b, l:iw - r, :] == mask)
            # _
            dA[:, t:ih - b, l:iw - r, :] += block * mask

    dA = layer.bc['dA'] = dA

    return dA
