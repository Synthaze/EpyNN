#EpyNN/nnlibs/flatten/backward.py
import numpy as np


def flatten_backward(layer,dA):

    # Cache dX (current) from dA (prev)
    dX = layer.bc['dX'] = dA

    # Cache dA (current) from dX (current)
    dA = layer.bc['dA'] = np.reshape(dX, layer.fs['X'])

    return dA
