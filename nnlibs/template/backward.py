# EpyNN/nnlibs/template/backward.py


def initialize_backward(layer, dA):
    """Backward cache initialization.

    :param layer: An instance of template layer.
    :type layer: :class:`nnlibs.template.models.Template`

    :param dA: Output of backward propagation from next layer.
    :type dA: :class:`numpy.ndarray`

    :return: Input of backward propagation for current layer.
    :rtype: :class:`numpy.ndarray`
    """
    dX = layer.bc['dX'] = dA

    return dX


def template_backward(layer, dA):
    """Backward propagate error to previous layer.
    """
    # (1) Initialize cache
    dX = initialize_backward(layer, dA)

    # (2) Pass backward
    dA = layer.bc['dA'] = dX

    return dA    # To previous layer
