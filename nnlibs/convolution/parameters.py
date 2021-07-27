# EpyNN/nnlibs/conv/parameters.py
# Standard library imports
import math
# Related third party imports
import numpy as np


def convolution_compute_shapes(layer, A):
    """Compute shapes for Convolution layer object

    :param layer: An instance of the :class:`nnlibs.convolution.models.Convolution`
    :type layer: class:`nnlibs.convolution.models.Convolution`
    """

    X = A.T

    layer.fs['X'] = X.shape

    layer.d['im'], layer.d['ih'], layer.d['iw'], layer.d['id'] = layer.fs['X']

    layer.fs['W'] = (layer.d['w'], layer.d['w'], layer.d['d'], layer.d['n'])
    layer.fs['b'] = (1, 1, 1, layer.d['n'])

    n_rows = layer.d['ih'] - layer.d['w'] + 1
    n_rows = min(layer.d['w'], n_rows)
    n_rows /= layer.d['s']

    n_cols = layer.d['ih'] - layer.d['w'] + 1
    n_cols = min(layer.d['w'], n_cols)
    n_cols /= layer.d['s']

    layer.d['R'] = math.ceil(n_rows)
    layer.d['C'] = math.ceil(n_cols)

    z_height = ((layer.d['ih'] - layer.d['w']) / layer.d['s']) + 1
    z_width = ((layer.d['iw'] - layer.d['w']) / layer.d['s']) + 1

    layer.d['zh'] = int(z_height)
    layer.d['zw'] = int(z_width)

    layer.fs['Z'] = (layer.d['im'], layer.d['zh'], layer.d['zw'], layer.d['n'])

    return None


def convolution_initialize_parameters(layer):
    """Dummy function - Initialize parameters for Convolution layer object

    :param layer: An instance of the :class:`nnlibs.convolution.models.Convolution`
    :type layer: class:`nnlibs.convolution.models.Convolution`
    """

    layer.p['W'] = layer.initialization(layer.fs['W'], rng=layer.np_rng)
    layer.p['b'] = np.zeros(layer.fs['b'])

    return None


def convolution_compute_gradients(layer):
    """Dummy function - Update weight and bias gradients for Convolution layer object

    :param layer: An instance of the :class:`nnlibs.convolution.models.Convolution`
    :type layer: class:`nnlibs.convolution.models.Convolution`
    """

    for parameter in layer.p.keys():
        gradient = 'd' + parameter
        layer.g[gradient] = np.zeros_like(layer.p[parameter])

    for t in range(layer.d['R']):

        for l in range(layer.d['C']):

            dW_block = layer.bc['dXb'][t][l] * layer.fc['Xb'][t][l]

            dW_block = np.sum(dW_block,axis=(2,1,0))

            layer.g['dW'] += dW_block

            db_block = np.sum(dW_block,axis=(2,1,0),keepdims=True)

            layer.g['db'] += db_block

    return None


def convolution_update_parameters(layer):
    """Dummy function - Update parameters for Convolution layer object

    :param layer: An instance of the :class:`nnlibs.convolution.models.Convolution`
    :type layer: class:`nnlibs.convolution.models.Convolution`
    """

    for gradient in layer.g.keys():

        parameter = gradient[1:]

        layer.p[parameter] -= layer.lrate[layer.e] * layer.g[gradient]

    return None
