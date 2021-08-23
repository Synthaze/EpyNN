# EpyNN/nnlibs/pooling/parameters.py
# Related third party imports
import numpy as np


def pooling_compute_shapes(layer, A):
    """Compute forward shapes and dimensions for layer.
    """
    X = A    # Input of current layer of shape (m, ih, iw, n)

    layer.fs['X'] = X.shape    # (m, ih, iw, n)

    layer.d['m'] = layer.fs['X'][0]     #
    layer.d['ih'] = layer.fs['X'][1]    #
    layer.d['iw'] = layer.fs['X'][2]    #
    layer.d['n'] = layer.fs['X'][3]     #

    return None


def pooling_initialize_parameters(layer):
    """Initialize parameters for layer.
    """
    # No parameters to initialize for Pooling layer

    return None


def pooling_compute_gradients(layer):
    """Compute gradients with respect to weight and bias for layer.
    """
    # No gradients to compute for Pooling layer

    return None


def pooling_update_parameters(layer):
    """Update parameters for layer.
    """
    # No parameters to update for Pooling layer

    return None
