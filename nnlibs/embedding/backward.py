# EpyNN/nnlibs/embedding/backward.py
# Related third party imports
import numpy as np


def embedding_backward(layer, dA):
    """

    :param layer: An instance of the :class:`nnlibs.embedding.models.Embedding`
    :type layer: class:`nnlibs.embedding.models.Embedding`

    :param dA:
    :type dA: class:`numpy.ndarray`
    """

    dX = initialize_backward(layer, dA)

    dA = layer.bc['dA'] = dX

    return dA


def initialize_backward(layer, dA):

    dX = layer.bc['dX'] = dA

    return dX
