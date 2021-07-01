#EpyNN/nnlibs/pool/forward.py
import numpy as np
import math


def pooling_forward(layer,A):

    # Cache X (current) from A (prev)
    X = layer.fc['X'] = A
    #im, ih, iw, id = layer.fs['X'] = X.shape
    id, iw, ih, im = layer.fs['X'] = X.shape

    layer.n_rows = math.ceil(min(layer.d['fw'], ih - layer.d['fw'] + 1) / layer.d['s'])
    layer.n_cols = math.ceil(min(layer.d['fw'], iw - layer.d['fw'] + 1) / layer.d['s'])

    z_h = int(((ih - layer.d['fw']) / layer.d['s']) + 1)
    z_w = int(((iw - layer.d['fw']) / layer.d['s']) + 1)

    Z = np.empty((im, z_h, z_w, id))

    # Loop through image rows
    for t in range(layer.n_rows):

        b = ih - (ih - t) % layer.d['fw']

        Z_cols = np.empty((im, int((b - t) / layer.d['fw']), z_w, id))

        # Loop through row columns
        for i in range(layer.n_cols):

            # _
            l = i * layer.d['s']
            r = iw - (iw - l) % layer.d['fw']

            # _
            block = X.T[:, t:b, l:r, :]

            # _
            block = np.array(np.split(block, (r - l) / layer.d['fw'], 2))
            block = np.array(np.split(block, (b - t) / layer.d['fw'], 2))

            # _
            block = layer.pool(block, 4)
            block = layer.pool(block, 3)

            # _
            block = np.moveaxis(block, 0, 2)
            block = np.moveaxis(block, 0, 2)

            # _
            Z_cols[:, :, i::layer.n_cols, :] = block

        # _
        Z[:, t * layer.d['s'] ::layer.n_rows, :, :] = Z_cols

    A = layer.fc['Z'] = layer.fc['A'] = Z

    return A
