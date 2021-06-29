#EpyNN/nnlibs/dropout/backward.py
import numpy as np


def dropout_backward(layer,dA):

    dA = np.multiply(dA,layer.D)

    dA = dA / layer.k

    return dA
