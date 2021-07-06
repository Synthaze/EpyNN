#EpyNN/nnlibs/conv/backward.py
import nnlibs.conv.parameters as cp

import numpy as np


def convolution_backward(layer,dA):

    dX, im, ih, iw, id, dA = cp.init_backward(layer,dA)

    # Loop through image rows
    for t in range(layer.n_rows):

        row = dX[:, t::layer.n_rows, :, :]

        # Loop through row columns
        for l in range(layer.n_cols):

            # region of X and dZ for this block
            b = (ih - t * layer.d['s']) % layer.d['fw']
            # region of X and dZ for this block
            r = (iw - l * layer.d['s']) % layer.d['fw']

            # block = corresponding region of dA
            block = row[:, :, l * layer.d['s']::layer.n_cols, :]
            # Axis for channels, rows, columns
            block = np.expand_dims(block, 3)
            block = np.expand_dims(block, 3)
            block = np.expand_dims(block, 3)

            layer.block = block

            # Update gradients
            cp.update_gradients(layer,t,l)

            dA_block = block * layer.p['W']
            dA_block = np.sum(dA_block, 6)
            dA_block = np.reshape(dA_block, (im, ih - b - t, iw - r - l, id))

            dA[:, t:ih - b, l:iw - r, :] += dA_block

    dA = layer.bc['dA'] = cp.restore_padding(layer,dA)

    return dA
