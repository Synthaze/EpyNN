# EpyNN/epynn/dropout/paremeters.py
# Related third party imports
import numpy as np


def dropout_compute_shapes(layer, A):
    """Compute forward shapes and dimensions from input for layer.
    """
    X = A    # Input of current layer

    layer.fs['X'] = X.shape    # (m, .. )

    layer.d['m'] = layer.fs['X'][0]          # Number of samples  (m)
    layer.d['n'] = X.size // layer.d['m']    # Number of features (n)

    # Shape for dropout mask
    layer.fs['D'] = [1 if ax in layer.d['a'] else layer.fs['X'][ax]
                     for ax in range(X.ndim)]

    return None


def dropout_initialize_parameters(layer):
    """Initialize trainable parameters from shapes for layer.
    """
    # No parameters to initialize for Dropout layer

    return None


def dropout_compute_gradients(layer):
    """Compute gradients with respect to weight and bias for layer.
    """
    # No gradients to update for Dropout layer

    return None


def dropout_update_parameters(layer):
    """Update parameters from gradients for layer.
    """
    # No parameters to update for Dropout layer

    return None
