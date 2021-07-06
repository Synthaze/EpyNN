#EpyNN/nnlibs/dropout/forward.py
import nnlibs.dropout.parameters as dp

import numpy as np


def dropout_forward(layer,A):

    X = dp.init_forward(layer,A)

    D = dp.init_mask(layer)

    A = np.multiply(X,D)

    A /= layer.k

    A = layer.fc['A'] = A

    return A
