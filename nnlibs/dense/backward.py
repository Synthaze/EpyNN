# EpyNN/nnlibs/dense/backward.py
# Related third party imports
import numpy as np


def initialize_backward(layer, dA):
    """Compute shapes for Dense layer object

    :param layer: An instance of the :class:`nnlibs.dense.models.Dense`
    :type layer: class:`nnlibs.dense.models.Dense`
    """

    dX = layer.bc['dX'] = dA

    return dX


def dense_backward(layer, dA):
    """Compute shapes for Dense layer object

    :param layer: An instance of the :class:`nnlibs.dense.models.Dense`
    :type layer: class:`nnlibs.dense.models.Dense`
    """

    dX = initialize_backward(layer, dA)

    dZ = layer.bc['dZ'] = dX * layer.activate(layer.fc['Z'], deriv=True)

    dA = layer.bc['dA'] = np.dot(layer.p['W'].T, dZ)

    return dA
