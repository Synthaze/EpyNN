# EpyNN/nnlibs/conv/parameters.py
# Standard library imports
import math

# Related third party imports
import numpy as np


def convolution_compute_shapes(layer, A):
    """Compute forward shapes and dimensions for layer.
    """
    X = A    # Input of current layer of shape (m , h, w, d)

    layer.fs['X'] = X.shape

    dims = ['m', 'ih', 'iw', 'd']

    layer.d.update({d:i for d,i in zip(dims, layer.fs['X'])})

    layer.fs['W'] = (layer.d['w'], layer.d['w'], layer.d['d'], layer.d['n'])
    layer.fs['b'] = (1, 1, 1, layer.d['n'])

    oh = layer.d['oh'] = math.floor((layer.d['ih'] - layer.d['w'] + 2 * layer.d['p']) / layer.d['s']) + 1
    ow = layer.d['ow'] = math.floor((layer.d['iw'] - layer.d['w'] + 2 * layer.d['p']) / layer.d['s']) + 1

    layer.fs['Z'] = (layer.d['m'], layer.d['oh'], layer.d['ow'], layer.d['n'])

    return None


def convolution_initialize_parameters(layer):
    """Initialize parameters for layer.
    """
    layer.p['W'] = layer.initialization(layer.fs['W'], rng=layer.np_rng)
    layer.p['b'] = np.zeros(layer.fs['b'])

    return None


def convolution_compute_gradients(layer):
    """Compute gradients with respect to weight and bias for layer.
    """
    # Gradients initialization with respect to parameters
    for parameter in layer.p.keys():
        gradient = 'd' + parameter
        layer.g[gradient] = np.zeros_like(layer.p[parameter])

    X = layer.fc['X']
    dX = layer.bc['dX']

    for i in range(layer.d['m']):

        for h in range(layer.d['oh']):
            ih1 = h * layer.d['s']
            ih2 = ih1 + layer.d['w']

            for w in range(layer.d['ow']):
                iw1 = w * layer.d['s']
                iw2 = iw1 + layer.d['w']

                for n in range(layer.d['n']):
                    layer.g['dW'][:, :, :, n] += X[i, ih1:ih2, iw1:iw2, :] * dX[i, h, w, n]
                    layer.g['db'][:, :, :, n] += dX[i, h, w, n]

    return None


def convolution_update_parameters(layer):
    """Update parameters for layer.
    """
    for gradient in layer.g.keys():
        parameter = gradient[1:]
        #
        layer.p[parameter] -= layer.lrate[layer.e] * layer.g[gradient]

    return None
