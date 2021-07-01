#EpyNN/nnlibs/conv/forward.py
import nnlibs.conv.parameters as cp

import numpy as np
import math


def convolution_forward(layer,A):

    # Cache X (current) from A (prev)
    X = layer.fc['X'] = A.T
    im, ih, iw, id = layer.fs['X'] = X.shape

    # Init layer parameters
    if layer.init == True:
        cp.init_params(layer)

    layer.n_rows = math.ceil(min(layer.d['fw'], ih - layer.d['fw'] + 1) / layer.d['s'])
    layer.n_cols = math.ceil(min(layer.d['fw'], iw - layer.d['fw'] + 1) / layer.d['s'])

    z_h = int(((ih - layer.d['fw']) / layer.d['s']) + 1)
    z_w = int(((iw - layer.d['fw']) / layer.d['s']) + 1)

    Z = np.empty((im, z_h, z_w, layer.d['nf']))

    layer.X_blocks = []

    # Loop through image rows
    for t in range(layer.n_rows):

        layer.X_blocks.append([])

        b = ih - (ih - t) % layer.d['fw']

        cols = np.empty((im, int((b - t) / layer.d['fw']), z_w, layer.d['nf']))

        # Loop through row columns
        for i in range(layer.n_cols):

            # _
            l = i * layer.d['s']; r = iw - (iw - l) % layer.d['fw']

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
            block = block * layer.p['W']

            # _
            block = np.sum(block, 5)
            block = np.sum(block, 4)
            block = np.sum(block, 3)

            # _
            cols[:, :, i::layer.n_cols, :] = block

        # _
        Z[:, t * layer.d['s'] ::layer.n_rows, :, :] = cols

    # _
    Z = layer.fc['Z'] = Z + layer.p['b']

    # _
    A = layer.fc['A'] = layer.activate(Z)

    return A
