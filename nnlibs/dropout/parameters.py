#EpyNN/nnlibs/dropout/paremeters.py

import numpy as np

def init_mask(layer):

    D = layer.np.random(layer.fs['D'])

    D = layer.fc['D'] = ( D < layer.k )

    return D


def init_forward(layer,A):

    # Cache X (current) from A (prev)
    X = layer.fc['X'] = A
    layer.fs['X'] = layer.fs['D'] = layer.fc['X'].shape

    return X


def init_backward(layer,dA):

    # Cache dX (current) from dA (prev)
    dX = layer.bc['dX'] = dA

    return dX
