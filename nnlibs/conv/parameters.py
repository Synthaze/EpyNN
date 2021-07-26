#EpyNN/nnlibs/conv/parameters.py
import nnlibs.commons.maths as cm

import numpy as np
import math


def set_activation(layer):

    args = layer.activation
    layer.activate, layer.derivative = args[0], cm.get_derivative(args[0])

    return None


def init_shapes(layer):

    fw = layer.d['fw'] = layer.f_width
    nf = layer.d['nf'] = layer.n_filters
    s = layer.d['s'] = layer.stride
    p = layer.d['p'] = layer.padding
    d = layer.d['d'] = layer.depth

    layer.fs['W'] = ( fw, fw, d, nf)
    layer.fs['b'] = ( 1, 1, 1, nf )

    return None


def init_forward(layer,A):

    # Cache X (current) from A (prev)
    X = layer.fc['X'] = A.T
    im, ih, iw, id = layer.fs['X'] = X.shape

    layer.n_rows = math.ceil(min(layer.d['fw'], ih - layer.d['fw'] + 1) / layer.d['s']) #
    layer.n_cols = math.ceil(min(layer.d['fw'], iw - layer.d['fw'] + 1) / layer.d['s']) #

    z_h = int(((ih - layer.d['fw']) / layer.d['s']) + 1)
    z_w = int(((iw - layer.d['fw']) / layer.d['s']) + 1)

    Z = np.empty((im, z_h, z_w, layer.d['nf']))

    layer.X_blocks = []

    return X, im, ih, iw, id, z_h, z_w, Z


def init_backward(layer,dA):

    # Cache dX (current) from dZ (prev)
    dX = layer.bc['dX'] = dA
    im, ih, iw, id = layer.fs['X']

    dA = np.zeros(layer.fs['X'])

    return dX, im, ih, iw, id, dA


def init_params(layer):

    layer.p['W'] = layer.initialization(layer.fs['W'])
    layer.p['b'] = np.zeros(layer.fs['b'])

    layer.init = False

    return None


def update_gradients(layer,t,l):

    dW_block = layer.block * layer.X_blocks[t][l]

    dW_block = np.sum(dW_block,axis=(2,1,0))

    layer.g['dW'] += dW_block

    db_block = np.sum(dW_block,axis=(2,1,0),keepdims=True)
#    db_block = np.sum(dW_block, 2, keepdims=True)
#    db_block = np.sum(db_block, 1, keepdims=True)
#    db_block = np.sum(db_block, 0, keepdims=True)

    layer.g['db'] += db_block

    return None


def restore_padding(layer,dA):

    if layer.d['p'] > 0:
        p = layer.d['p']
        dA = dA[:, p:-p, p:-p, :]

    return dA
