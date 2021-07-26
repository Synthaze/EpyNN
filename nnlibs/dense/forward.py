# EpyNN/nnlibs/dense/forward.py
# Related third party imports
import numpy as np


def dense_forward(layer, A):
    """

    :param layer: An instance of the :class:`nnlibs.dense.models.Dense`
    :type layer: class:`nnlibs.dense.models.Dense`
    """

    X = initialize_forward(layer, A)

    Z = layer.fc['Z'] = np.dot(layer.p['W'], X) + layer.p['b']

    A = layer.fc['A'] = layer.activate(Z)

    return A


def initialize_forward(layer, A):

    X = layer.fc['X'] = A

    return X
