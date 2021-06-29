#EpyNN/nnlibs/flatten/forward.py
import numpy as np


#@log_function
def flatten_forward(layer,A):

    layer.X = A

    layer.s['X'] = layer.X.shape

    layer.s['A'] = (int(layer.X.size / layer.s['X'][-1]),layer.s['X'][-1])

    layer.A = np.reshape(layer.X,layer.s['A'])

    return layer.A
