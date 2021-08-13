# EpyNN/nnlibs/flatten/parameters.py


def flatten_compute_shapes(layer, A):
    """Compute forward shapes and dimensions for layer.
    """
    X = A    # Input of current layer of shape (m .. n)

    layer.fs['X'] = X.shape

    layer.d['m'] = layer.fs['X'][0]    # Number of samples (m)
    layer.d['n'] = layer.fs['X'][1]    # @

    for i in range(2, X.ndim):
        layer.d['n'] *= layer.fs['X'][i]

    #
    layer.fs['A'] = (layer.d['m'], int(X.size / layer.d['m']))

    return None


def flatten_initialize_parameters(layer):
    """Initialize parameters for layer.
    """
    # No parameters to initialize for Flatten layer

    return None


def flatten_compute_gradients(layer):
    """Compute gradients with respect to weight and bias for layer.
    """
    # No gradients to update for Flatten layer

    return None


def flatten_update_parameters(layer):
    """Update parameters for layer.
    """
    # No parameters to update for Flatten layer

    return None
