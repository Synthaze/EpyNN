# EpyNN/nnlibs/template/parameters.py


def template_compute_shapes(layer, A):
    """Compute forward shapes and dimensions for cells and layer.
    """
    X = A    # Input of current layer of shape (m .. n)

    layer.d['n'] = X.shape[1:]    # Features
    layer.d['m'] = X.shape[0]     # Number of samples (m)

    return None


def template_initialize_parameters(layer):
    """Initialize parameters for layer.
    """
    # No parameters to initialize for Template layer

    return None


def template_compute_gradients(layer):
    """Compute gradients with respect to weight and bias for layer.
    """
    # No gradients to update for Template layer

    return None


def template_update_parameters(layer):
    """Update parameters for layer.
    """
    # No parameters to update for Template layer

    return None
