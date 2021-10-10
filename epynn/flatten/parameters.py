# EpyNN/epynn/flatten/parameters.py


def flatten_compute_shapes(layer, A):
    """Compute forward shapes and dimensions from input for layer.
    """
    X = A    # Input of current layer

    layer.fs['X'] = X.shape    # (m, ...)

    layer.d['m'] = layer.fs['X'][0]          # Number of samples  (m)
    layer.d['n'] = X.size // layer.d['m']    # Number of features (n)

    # Shape for output of forward propagation
    layer.fs['A'] = (layer.d['m'], layer.d['n'])

    return None


def flatten_initialize_parameters(layer):
    """Initialize trainable parameters from shapes for layer.
    """
    # No parameters to initialize for Flatten layer

    return None


def flatten_compute_gradients(layer):
    """Compute gradients with respect to weight and bias for layer.
    """
    # No gradients to compute for Flatten layer

    return None


def flatten_update_parameters(layer):
    """Update parameters from gradients for layer.
    """
    # No parameters to update for Flatten layer

    return None
