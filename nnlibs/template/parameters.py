# EpyNN/nnlibs/template/parameters.py


def template_compute_shapes(layer, A):
    """Compute shapes for Template layer object

    :param layer: An instance of the :class:`nnlibs.template.models.Template`
    :type layer: class:`nnlibs.template.models.Template`
    """

    X = A

    return None


def template_initialize_parameters(layer):
    """Dummy function - Initialize parameters for Template layer object

    :param layer: An instance of the :class:`nnlibs.template.models.Template`
    :type layer: class:`nnlibs.template.models.Template`
    """

    # No parameters to initialize for Template layer

    return None


def template_compute_gradients(layer):
    """Dummy function - Update weight and bias gradients for Template layer object

    :param layer: An instance of the :class:`nnlibs.template.models.Template`
    :type layer: class:`nnlibs.template.models.Template`
    """

    # No gradients to update for Template layer

    return None


def template_update_parameters(layer):
    """Dummy function - Update parameters for Template layer object

    :param layer: An instance of the :class:`nnlibs.template.models.Template`
    :type layer: class:`nnlibs.template.models.Template`
    """

    # No parameters to update for Template layer

    return None
