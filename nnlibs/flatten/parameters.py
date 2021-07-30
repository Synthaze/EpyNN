# EpyNN/nnlibs/flatten/parameters.py


def flatten_compute_shapes(layer, A):
    """Compute shapes for Flatten layer object

    :param layer: An instance of the :class:`nnlibs.flatten.models.Flatten`
    :type layer: class:`nnlibs.flatten.models.Flatten`
    """

    X = A

    layer.fs['X'] = X.shape

    layer.d['m'] = layer.fs['X'][-1]
    layer.d['n'] = layer.fs['X'][0] * layer.fs['X'][1]

    layer.fs['A'] = (int(X.size / X.shape[-1]), X.shape[-1])

    return None


def flatten_initialize_parameters(layer):
    """Dummy function - Initialize parameters for Flatten layer object

    :param layer: An instance of the :class:`nnlibs.flatten.models.Flatten`
    :type layer: class:`nnlibs.flatten.models.Flatten`
    """

    # No parameters to initialize for Flatten layer

    return None


def flatten_compute_gradients(layer):
    """Dummy function - Update weight and bias gradients for Flatten layer object

    :param layer: An instance of the :class:`nnlibs.flatten.models.Flatten`
    :type layer: class:`nnlibs.flatten.models.Flatten`
    """

    # No gradients to update for Flatten layer

    return None


def flatten_update_parameters(layer):
    """Dummy function - Update parameters for Flatten layer object

    :param layer: An instance of the :class:`nnlibs.flatten.models.Flatten`
    :type layer: class:`nnlibs.flatten.models.Flatten`
    """

    # No parameters to update for Flatten layer

    return None
