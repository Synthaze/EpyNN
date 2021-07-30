# EpyNN/nnlibs/flatten/backward.py
# Related third party imports
import numpy as np


def flatten_backward(layer, dA):
    """

    :param layer: An instance of the :class:`nnlibs.flatten.models.Flatten`
    :type layer: class:`nnlibs.flatten.models.Flatten`

    :param dA:
    :type dA: class:`numpy.ndarray`
    """

    dX = initialize_backward(layer, dA)

    dA = layer.bc['dA'] = np.reshape(dX, layer.fs['X'])

    return dA


def initialize_backward(layer, dA):

    dX = layer.bc['dX'] = dA

    return dX
