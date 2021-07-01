#EpyNN/nnlibs/dense/backward.py
import nnlibs.dense.parameters as dp

import numpy as np


def dense_backward(layer,dA):

    # Cache dX (current) from dA (prev)
    dX = layer.bc['dX'] = dA

    # Cache dZ (current) from dX (prev)
    dZ = layer.bc['dZ'] = layer.derivative( dX, layer.fc['Z'] )

    # Cache dA (current) from dZ (current)
    dA = layer.bc['dA'] = np.dot( layer.p['W'].T, dZ )

    # Update layer gradients
    dp.update_grads(layer)

    return dA
