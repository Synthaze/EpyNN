#EpyNN/nnlibs/pool/parameters.py
import math

import numpy as np


def init_shapes(layer):
    ### Set layer dictionaries values
    ## Dimensions
    layer.d['fw'] = layer.f_width
    layer.d['s'] = layer.stride

    return None


def init_forward(layer,A):

    # Cache X (current) from A (prev)
    X = layer.fc['X'] = A
    #im, ih, iw, id = layer.fs['X'] = X.shape
    id, iw, ih, im = layer.fs['X'] = X.shape

    layer.n_rows = math.ceil(min(layer.d['fw'], ih - layer.d['fw'] + 1) / layer.d['s'])
    layer.n_cols = math.ceil(min(layer.d['fw'], iw - layer.d['fw'] + 1) / layer.d['s'])

    z_h = int(((ih - layer.d['fw']) / layer.d['s']) + 1)
    z_w = int(((iw - layer.d['fw']) / layer.d['s']) + 1)

    Z = np.empty((im, z_h, z_w, id))

    return X, id, iw, ih, im, z_h, z_w, Z


def init_backward(layer,dA):

    # Cache X (current) from A (prev)
    dX = layer.bc['dX'] = dA
    im, ih, iw, id = layer.fs['X']

    dA = np.zeros( layer.fs['X'] )

    return dX, im, ih, iw, id, dA
