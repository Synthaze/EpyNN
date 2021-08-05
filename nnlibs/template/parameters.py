# EpyNN/nnlibs/template/parameters.py


def template_compute_shapes(layer, A):
    """Compute dimensions and forward shapes for layer.
    """
    X = A    # Input of current layer of shape (m .. n)

    layer.d['m'] = X.shape[0]     # Number of samples (m)
    layer.d['n'] = X.shape[1:]    # Sample features (.. n)

    return None


def template_initialize_parameters(layer):
    """Initialize parameters from shapes for layer.
    """
    # No parameters to initialize for Template layer

    return None


def template_compute_gradients(layer):
    """Compute gradients of the cost with respect to weight and bias for layer.
    """
    # No gradients to compute for Template layer

    return None


def template_update_parameters(layer):
    """Update parameters from gradients and learning rate for layer.
    """
    # No parameters to update for Template layer

    return None
