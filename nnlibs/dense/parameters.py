#EpyNN/nnlibs/dense/parameters.py
import nnlibs.commons.maths as cm

import nnlibs.meta.parameters as mp

import numpy as np


def set_activation(layer):

    args = layer.activation
    layer.activate, layer.derivative = args[0], cm.get_derivative(args[0])

    return None


def init_shapes(layer,layer_size):
    ### Set layer dictionaries values
    ## Dimensions
    # Layer size
    layer.d['d'] = layer_size

    return None


def init_params(layer):

    layer.fs['W'] = ( layer.d['d'], layer.fs['X'][0] )

    layer.fs['b'] = ( layer.d['d'], 1 )

    layer.p['W'] = np.random.randn( *layer.fs['W'] )
    layer.p['W'] /= np.sqrt(layer.fs['W'][1])

    layer.p['b'] = np.zeros( layer.fs['b'] )

    layer.init = False

    return None


def update_grads(layer):

    m = layer.m = layer.fs['X'][-1]

    X = layer.fc['X']

    layer.g['dW'] = 1./ m * np.dot(layer.bc['dZ'],X.T)

    layer.g['db'] = 1./ m * np.sum(layer.bc['dZ'],axis=1,keepdims=True)

    return None
