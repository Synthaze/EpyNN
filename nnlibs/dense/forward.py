#EpyNN/nnlibs/dense/forward.py
import nnlibs.dense.parameters as dp

import numpy as np


#@log_function
def dense_forward(layer,A):

    layer.X = A

    if layer.init == True:
        dp.init_params(layer)

    layer.Z = np.dot(layer.p['W'],layer.X) + layer.p['b']

    layer.s['Z'] = layer.Z.shape

    layer.A = layer.activate(layer.Z)

    return layer.A
