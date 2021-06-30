#EpyNN/nnlibs/dropout/forward.py
import nnlibs.dropout.parameters as dp

import numpy as np


def dropout_forward(layer,A):

    layer.X = A

    layer.s['X'] = layer.X.shape
    layer.s['D'] = layer.s['X']

    dp.init_mask(layer)

    layer.A = np.multiply(layer.X,layer.D)

    layer.A /= layer.k

    return layer.A
