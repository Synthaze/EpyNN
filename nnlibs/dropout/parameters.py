# EpyNN/nnlibs/dropout/paremeters.py
# Related third party imports
import numpy as np


def dropout_compute_shapes(layer, A):
    """Compute forward shapes and dimensions for layer.
    """
    X = A    # Input of current layer of shape (m .. n)

    layer.fs['X'] = layer.fs['D'] = X.shape

    layer.d['m'] = layer.fs['X'][0]        # Number of samples (m)
    layer.d['n'] = X.size // layer.d['m']  # @

    return None


def dropout_initialize_parameters(layer):
    """Initialize parameters for layer.
    """
    # No parameters to initialize for Dropout layer

    return None


def dropout_compute_gradients(layer):
    """Compute gradients with respect to weight and bias for layer.
    """
    # No gradients to update for Dropout layer

    return None


def dropout_update_parameters(layer):
    """Update parameters for layer.
    """
    # No parameters to update for Dropout layer

    return None
