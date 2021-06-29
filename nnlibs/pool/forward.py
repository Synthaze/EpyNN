#EpyNN/nnlibs/pool/forward.py
import numpy as np
import math


def pooling_forward(layer,A):

    layer.X = A

    layer.s['X'] = layer.X.shape

    id, iw, ih, im = layer.s['X']

    layer.n_rows = math.ceil(min(layer.d['fw'], ih - layer.d['fw'] + 1) / layer.d['s'])
    layer.n_cols = math.ceil(min(layer.d['fw'], iw - layer.d['fw'] + 1) / layer.d['s'])

    z_h = int(((ih - layer.d['fw']) / layer.d['s']) + 1)
    z_w = int(((iw - layer.d['fw']) / layer.d['s']) + 1)

    layer.s['Z'] = (im, z_h, z_w, id)

    layer.Z = np.empty(layer.s['Z'])

    for t in range(layer.n_rows):

        b = ih - (ih - t) % layer.d['fw']

        Z_cols = np.empty((im, int((b - t) / layer.d['fw']), z_w, id))

        for i in range(layer.n_cols):

            l = i * layer.d['s']
            r = iw - (iw - l) % layer.d['fw']

            block = layer.X.T[:, t:b, l:r, :]

            block = np.array(np.split(block, (r - l) / layer.d['fw'], 2))
            block = np.array(np.split(block, (b - t) / layer.d['fw'], 2))

            block = layer.pool(block, 4)
            block = layer.pool(block, 3)

            block = np.moveaxis(block, 0, 2)
            block = np.moveaxis(block, 0, 2)

            Z_cols[:, :, i::layer.n_cols, :] = block

        layer.Z[:, t * layer.d['s'] ::layer.n_rows, :, :] = Z_cols

    return layer.Z
