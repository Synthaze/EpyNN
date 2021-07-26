#EpyNN/nnlibs/conv/forward.py
import nnlibs.conv.parameters as cp

import numpy as np


def convolution_forward(layer,A):

    X, im, ih, iw, id, z_h, z_w, Z = cp.init_forward(layer,A)

    # Init layer parameters
    if layer.init == True:
        cp.init_params(layer)

    # Loop through image rows
    for t in range(layer.n_rows):

        layer.X_blocks.append([])

        b = ih - (ih - t) % layer.d['fw']

        cols = np.empty((im, int((b - t) / layer.d['fw']), z_w, layer.d['nf']))

        # Loop through row columns
        for i in range(layer.n_cols):

            # _
            l = i * layer.d['s']

            r = iw - (iw - l) % layer.d['fw']

            # _
            block = X[:, t:b, l:r, :]

            # _
            block = np.array(np.split(block, (r - l) / layer.d['fw'], 2))
            block = np.array(np.split(block, (b - t) / layer.d['fw'], 2))

            # _
            block = np.moveaxis(block, 2, 0)

            # _
            block = np.expand_dims(block, 6)

            # _
            layer.X_blocks[t].append(block)

            # _
            block *= layer.p['W']

            # _
            block = np.sum(block,axis=5)
            block = np.sum(block,axis=4)
            block = np.sum(block,axis=3)

            # _
            cols[:, :, i::layer.n_cols, :] = block

        # _
        Z[:, t * layer.d['s'] ::layer.n_rows, :, :] = cols

    # _
    Z = layer.fc['Z'] = Z + layer.p['b']

    # _
    A = layer.fc['A'] = layer.activate(Z)

    return A
