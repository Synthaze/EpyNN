#EpyNN/nnlibs/dense/forward.py
import nnlibs.meta.parameters as mp

import nnlibs.dense.parameters as dp

import numpy as np


def dense_forward(layer,A):

    # Cache X (current) from A (prev)
    X = layer.fc['X'] = A
    layer.fs['X'] = layer.fc['X'].shape

    # Init layer parameters
    if layer.init == True:
        dp.init_params(layer)

    # Cache Z (current) from X (current)
    Z = layer.fc['Z'] = np.dot( layer.p['W'], X ) + layer.p['b']

    # Cache A (current) from Z (current)
    A = layer.fc['A'] = layer.activate(Z)

    return A
