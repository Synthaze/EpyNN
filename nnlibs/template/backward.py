# EpyNN/nnlibs/template/backward.py


def template_backward(layer, dA):
    """

    :param layer: An instance of the :class:`nnlibs.template.models.Template`
    :type layer: class:`nnlibs.template.models.Template`

    :param dA:
    :type dA: class:`numpy.ndarray`
    """

    dX = initialize_backward(layer, dA)

    dA = layer.bc['dA'] = dX

    return dA


def initialize_backward(layer, dA):

    dX = layer.bc['dX'] = dA

    return dX
