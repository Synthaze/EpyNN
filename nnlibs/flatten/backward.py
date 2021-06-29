#EpyNN/nnlibs/flatten/backward.py
import numpy as np


def flatten_backward(layer,dA):

    dA = np.reshape(dA, layer.s['X'])

    return dA
