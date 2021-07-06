#EpyNN/nnlibs/dropout/backward.py
import nnlibs.dropout.parameters as dp

import numpy as np


def dropout_backward(layer,dA):

    dX = dp.init_backward(layer,dA)

    dA = np.multiply(dX,layer.fc['D'])

    dA /= layer.k

    layer.bc['dA'] = dA

    return dA
