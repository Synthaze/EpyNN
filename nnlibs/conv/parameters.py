#EpyNN/nnlibs/conv/parameters.py
import nnlibs.commons.maths as cm

import numpy as np


def set_activation(layer):

    args = layer.activation
    layer.activate, layer.derivative = args[0], cm.get_derivative(args[0])

    return None


def init_shapes(layer,n_filters,f_width,depth,stride,padding):

    fw = layer.d['fw'] = f_width
    nf = layer.d['nf'] = n_filters
    s = layer.d['s'] = stride
    p = layer.d['p'] = padding
    d = layer.d['d'] = depth

    layer.fs['W'] = ( fw, fw, d, nf)
    layer.fs['b'] = ( 1, 1, 1, nf )

    return None


def init_params(layer):

    layer.p['W'] = np.random.random( layer.fs['W'] ) * 0.1
    layer.p['b'] = np.random.random( layer.fs['b'] ) * 0.01

    layer.init = False

    return None


def update_gradients(layer,t,l):

    dW_block = layer.block * layer.X_blocks[t][l]

    dW_block = np.sum(dW_block, 2)
    dW_block = np.sum(dW_block, 1)
    dW_block = np.sum(dW_block, 0)

    layer.g['dW'] += dW_block

    db_block = np.sum(dW_block, 2, keepdims=True)
    db_block = np.sum(db_block, 1, keepdims=True)
    db_block = np.sum(db_block, 0, keepdims=True)

    layer.g['db'] += db_block

    return None


def restore_padding(layer,dA):

    if layer.d['p'] > 0:
        p = layer.d['p']
        dA = dA[:, p:-p, p:-p, :]

    return dA
