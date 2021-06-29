#EpyNN/nnlibs/pool/backward.py
import numpy as np


def pooling_backward(layer,dA):

    im, ih, iw, id = layer.X.shape

    dA_prev = np.zeros(layer.X.shape)

    for t in range(layer.n_rows):

        mask_row = layer.Z[:, t::layer.n_rows, :, :]
        row = dA[:, t::layer.n_rows, :, :]

        for l in range(layer.n_cols):

            b = (ih - t * layer.d['s']) % layer.d['fw']
            r = (iw - l * layer.d['s']) % layer.d['fw']

            mask = mask_row[:, :, l * layer.d['s']::layer.n_cols, :]

            mask = layer.assemble_block(mask, t, b, l, r)

            block = row[:, :, l * layer.d['s']::layer.n_cols, :]

            block = layer.assemble_block(block, t, b, l, r)

            mask = (layer.X[:, t:ih - b, l:iw - r, :] == mask)

            dA_prev[:, t:ih - b, l:iw - r, :] += block * mask

    layer.g = {}

    return dA_prev
