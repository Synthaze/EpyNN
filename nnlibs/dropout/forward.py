#EpyNN/nnlibs/dropout/forward.py
import numpy as np

#@log_function
def dropout_forward(layer,A):

    D = np.random.rand(A.shape[0],A.shape[1])

    layer.D = ( D < layer.k )

    layer.X = A

    A = np.multiply(layer.X,layer.D)

    layer.A = A / layer.k

    return layer.A
