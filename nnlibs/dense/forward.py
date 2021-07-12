#EpyNN/nnlibs/dense/forward.py
import nnlibs.dense.parameters as dp

import numpy as np


def dense_forward(layer,A):
    """
    Description for function

    :ivar var1: initial value: par1
    :ivar var2: initial value: par2
    """

    X = dp.init_forward(layer,A)

    # Init layer parameters
    if layer.init == True:
        dp.init_params(layer)

    # Cache Z (current) from X (current)
    Z = layer.fc['Z'] = np.dot( layer.p['W'], X ) + layer.p['b']

    # Cache A (current) from Z (current)
    A = layer.fc['A'] = layer.activate(Z)

    return A
