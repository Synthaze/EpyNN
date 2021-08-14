# EpyNN/nnlibs/conv/parameters.py
# Standard library imports
import math

# Related third party imports
from nnlibs.commons.io import padding
import numpy as np


def convolution_compute_shapes(layer, A):
    """Compute forward shapes and dimensions for layer.
    """
    X = padding(A, layer.d['p'])    # Input of current layer

    layer.fs['X'] = X.shape    # (m , ih, iw, id)

    dims = ['m', 'ih', 'iw', 'id']

    layer.d.update({d: i for d, i in zip(dims, layer.fs['X'])})

    # Apply to X_block - W shape is (filter_width, filter_width, image_depth, n_filters)
    layer.fs['W'] = (layer.d['w'], layer.d['w'], layer.d['id'], layer.d['n'])
    layer.fs['b'] = (1, 1, 1, layer.d['n'])

    layer.d['oh'] = math.ceil(min(layer.d['w'], layer.d['ih'] - layer.d['w'] + 1) / layer.d['s'])
    layer.d['ow'] = math.ceil(min(layer.d['w'], layer.d['iw'] - layer.d['w'] + 1) / layer.d['s'])

    layer.d['zh'] = int(((layer.d['ih'] - layer.d['w']) / layer.d['s']) + 1)
    layer.d['zw'] = int(((layer.d['iw'] - layer.d['w']) / layer.d['s']) + 1)

    # Shape of cache for linear activation product Z
    layer.fs['Z'] = (layer.d['m'], layer.d['zh'], layer.d['zw'], layer.d['n'])

    return None


def convolution_initialize_parameters(layer):
    """Initialize parameters for layer.
    """
    # W, b - Linear activation X_block -> Z_block
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

    dX = layer.bc['dX']

    # Iterate over image rows
    for t in range(layer.d['oh']):
        #
        row = dX[:, t::layer.d['oh'], :, :]

        # Iterate over image columns
        for l in range(layer.d['ow']):

            #
            b = (layer.d['ih'] - t * layer.d['s']) % layer.d['w']
            r = (layer.d['iw'] - l * layer.d['s']) % layer.d['w']

            #
            block = row[:, :, l * layer.d['s']::layer.d['oh'], :]

            #
            block = np.expand_dims(block, axis=3)
            block = np.expand_dims(block, axis=3)
            block = np.expand_dims(block, axis=3)

            # Gradients with respect to W
            dW = block * layer.Xb[t][l]
            dW = np.sum(dW, axis=2)
            dW = np.sum(dW, axis=1)
            dW = np.sum(dW, axis=0)

            layer.g['dW'] += dW

            # Gradients with respect to b
            db = np.sum(dW, axis=2, keepdims=True)
            db = np.sum(db, axis=1, keepdims=True)
            db = np.sum(db, axis=0, keepdims=True)

            layer.g['db'] += db

    return None


def convolution_update_parameters(layer):
    """Update parameters for layer.
    """
    for gradient in layer.g.keys():
        parameter = gradient[1:]
        # Update is driven by learning rate and gradients
        layer.p[parameter] -= layer.lrate[layer.e] * layer.g[gradient]

    return None
