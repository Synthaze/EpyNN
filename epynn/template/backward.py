# EpyNN/epynn/template/backward.py


def initialize_backward(layer, dX):
    """Backward cache initialization.

    :param layer: An instance of template layer.
    :type layer: :class:`epynn.template.models.Template`

    :param dX: Output of backward propagation from next layer.
    :type dX: :class:`numpy.ndarray`

    :return: Input of backward propagation for current layer.
    :rtype: :class:`numpy.ndarray`
    """
    dA = layer.bc['dA'] = dX

    return dA


def template_backward(layer, dX):
    """Backward propagate error gradients to previous layer.
    """
    # (1) Initialize cache
    dA = initialize_backward(layer, dX)

    # (2) Pass backward
    dX = layer.bc['dX'] = dA

    return dX    # To previous layer
