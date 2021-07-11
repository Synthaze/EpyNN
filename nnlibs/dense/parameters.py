#EpyNN/nnlibs/dense/parameters.py
import nnlibs.commons.maths as cm

import nnlibs.meta.parameters as mp

import numpy as np


def set_activation(layer):

    args = layer.activation
    layer.activate, layer.derivative = args[0], cm.get_derivative(args[0])

    return None


def init_shapes(layer,nodes):
    ### Set layer dictionaries values
    ## Dimensions
    # Layer size
    layer.d['d'] = nodes

    dx = ( layer.d['d'], None )
    d1 = ( layer.d['d'], 1 )

    layer.fs['W'], layer.fs['b'] = dx, d1

    return None


def init_forward(layer,A):

    # Cache X (current) from A (prev)
    X = layer.fc['X'] = A
    layer.fs['X'] = layer.fc['X'].shape

    return X


def init_backward(layer,dA):

    # Cache dX (current) from dA (prev)
    dX = layer.bc['dX'] = dA

    return dX


def init_params(layer):

    dx = ( layer.fs['W'][0], layer.fs['X'][0] )

    layer.fs['W'] = dx

    layer.p['W'] = layer.initialization(layer.fs['W'])
    layer.p['b'] = np.zeros( layer.fs['b'] )

    layer.init = False

    return None


def update_grads(layer):

    m = layer.m = layer.fs['X'][-1]

    X = layer.fc['X']

    l1 = layer.l1 / m
    l2 = layer.l2 * layer.p['W'] / m

    layer.g['dW'] = 1./ m * np.dot(layer.bc['dZ'],X.T) + l1 + l2

    layer.g['db'] = 1./ m * np.sum(layer.bc['dZ'],axis=1,keepdims=True)

    return None
