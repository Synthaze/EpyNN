# EpyNN/nnlibs/dense/parameters.py
# Related third party imports
import numpy as np


def dense_compute_shapes(layer, A):
    """Compute shapes for Dense layer object

    :param layer: An instance of the :class:`nnlibs.dense.models.Dense`
    :type layer: class:`nnlibs.dense.models.Dense`
    """

    X = A

    layer.fs['X'] = X.shape

    layer.d['p'] = layer.fs['X'][0]
    layer.d['m'] = layer.fs['X'][1]

    nm = layer.fs['W'] = (layer.d['n'], layer.d['p'])
    n1 = layer.fs['b'] = (layer.d['n'], 1)

    return None


def dense_initialize_parameters(layer):
    """Initialize parameters for Dense layer object

    :param layer: An instance of the :class:`nnlibs.dense.models.Dense`
    :type layer: class:`nnlibs.dense.models.Dense`
    """

    layer.p['W'] = layer.initialization(layer.fs['W'], rng=layer.np_rng)
    layer.p['b'] = np.zeros(layer.fs['b'])

    return None


def dense_compute_gradients(layer):
    """Update weight and bias gradients for Dense layer object

    :param layer: An instance of the :class:`nnlibs.dense.models.Dense`
    :type layer: class:`nnlibs.dense.models.Dense`
    """

    # X - Input of forward propagation
    X = layer.fc['X']
    # dZ - Gradient of the cost with respect to the linear output of forward propagation (Z)
    dZ = layer.bc['dZ']

    # dW - Gradient of the cost with respect to weight (W)
    dW = layer.g['dW'] = 1./ layer.d['m'] * np.dot(dZ, X.T)
    # db - Gradient of the cost with respect to biais (b)
    db = layer.g['db'] = 1./ layer.d['m'] * np.sum(dZ, axis=1, keepdims=True)

    return None


def dense_update_parameters(layer):
    """Update parameters for Dense layer object

    :param layer: An instance of the :class:`nnlibs.dense.models.Dense`
    :type layer: class:`nnlibs.dense.models.Dense`
    """

    for gradient in layer.g.keys():

        parameter = gradient[1:]

        layer.p[parameter] -= layer.lrate[layer.e] * layer.g[gradient]

    return None
