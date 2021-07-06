#EpyNN/nnlibs/flatten/forward.py
import numpy as np


def flatten_forward(layer,A):

    # Cache X (current) from A (prev)
    X = layer.fc['X'] = A
    layer.fs['X'] = X.shape

    # Compute shape A (current) from X (current)
    layer.fs['A'] = ( int(X.size / X.shape[-1]), X.shape[-1] )

    # Cache A (current) from X (prev)
    A = layer.fc['A'] = np.reshape( X, layer.fs['A'] )
    
    return A
