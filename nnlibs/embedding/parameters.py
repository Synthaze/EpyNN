# EpyNN/nnlibs/embedding/parameters.py
# Related third party imports
import numpy as np


def embedding_compute_shapes(layer, A):
    """Compute forward shapes and dimensions for layer.
    """
    X = A    # Input of current layer of shape (m .. n)

    layer.fs['X'] = X.shape

    layer.d['m'] = layer.fs['X'][0]    # Number of samples (m)
    layer.d['n'] = layer.fs['X'][1]    # @

    return None


def embedding_initialize_parameters(layer):
    """Initialize parameters for layer.
    """
    # No parameters to initialize for Embedding layer

    return None


def embedding_compute_gradients(layer):
    """Compute gradients with respect to weight and bias for layer.
    """
    # No gradients to update for Embedding layer

    return None


def embedding_update_parameters(layer):
    """Update parameters for layer.
    """
    # No parameters to update for Embedding layer

    return None
