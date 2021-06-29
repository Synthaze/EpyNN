#EpyNN/nnlibs/dense/backward.py
import numpy as np


def dense_backward(layer,dA):

    m = dA.shape[1]

    dZ = layer.derivative(dA,layer.Z)

    layer.g['dW'] = 1./ m * np.dot(dZ,layer.X.T)

    layer.g['db'] = 1./ m * np.sum(dZ,axis=1,keepdims=True)

    dA_prev = np.dot(layer.p['W'].T,dZ)

    return dA_prev
