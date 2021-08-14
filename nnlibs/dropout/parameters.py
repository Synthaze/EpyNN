# EpyNN/nnlibs/dropout/paremeters.py


def dropout_compute_shapes(layer, A):
    """Compute forward shapes and dimensions for layer.
    """
    X = A    # Input of current layer

    layer.fs['X'] = layer.fs['D'] = X.shape  # (m .. n)

    layer.d['m'] = layer.fs['X'][0]        # Number of samples (m)
    layer.d['n'] = X.size // layer.d['m']  # Number of 1D features (n)

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
