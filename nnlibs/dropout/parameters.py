# EpyNN/nnlibs/dropout/paremeters.py
# Related third party imports
import numpy as np


def dropout_compute_shapes(layer, A):
    """Compute forward shapes and dimensions from input for layer.
    """
    X = A    # Input of current layer

    layer.fs['X'] = X.shape    # (m, .. )

    layer.d['m'] = layer.fs['X'][0]          # Number of samples  (m)

    # Shape for dropout mask
    layer.fs['D'] = [1 if i in layer.d['a'] else layer.fs['X'][i] for i in range(X.ndim)]

    # Scaling factor for signal
    layer.d['s'] = np.zeros(layer.fs['D']).size / X.size
    layer.d['s'] *= (1 - layer.d['d'])

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
