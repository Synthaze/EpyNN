#EpyNN/nnlibs/conv/forward.py
import nnlibs.conv.parameters as cp

import numpy as np
import math


def convolution_forward(layer,A):

    layer.X = A.T

    layer.s['X'] = layer.X.shape

    im, ih, iw, id = layer.s['X']

    if layer.init == True:
        cp.init_params(layer)

    layer.n_rows = math.ceil(min(layer.d['fw'], ih - layer.d['fw'] + 1) / layer.d['s'])

    layer.n_cols = math.ceil(min(layer.d['fw'], iw - layer.d['fw'] + 1) / layer.d['s'])

    z_h = int(((ih - layer.d['fw']) / layer.d['s']) + 1);

    z_w = int(((iw - layer.d['fw']) / layer.d['s']) + 1)

    layer.Z = np.empty((im, z_h, z_w, layer.d['n_f']))

    layer.X_blocks = []

    for t in range(layer.n_rows):

        layer.X_blocks.append([])

        b = ih - (ih - t) % layer.d['fw']

        cols = np.empty((im, int((b - t) / layer.d['fw']), z_w, layer.d['n_f']))

        for i in range(layer.n_cols):

            l = i * layer.d['s']; r = iw - (iw - l) % layer.d['fw']

            block = layer.X[:, t:b, l:r, :]

            block = np.array(np.split(block, (r - l) / layer.d['fw'], 2))
            block = np.array(np.split(block, (b - t) / layer.d['fw'], 2))

            block = np.moveaxis(block, 2, 0)

            block = np.expand_dims(block, 6)

            layer.X_blocks[t].append(block)

            block = block * layer.p['W']

            block = np.sum(block, 5)
            block = np.sum(block, 4)
            block = np.sum(block, 3)

            cols[:, :, i::layer.n_cols, :] = block

        layer.Z[:, t * layer.d['s'] ::layer.n_rows, :, :] = cols

    layer.Z += layer.p['b']

    layer.A = layer.activate(layer.Z)

    return layer.A
