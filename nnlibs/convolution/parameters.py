# EpyNN/nnlibs/convolution/parameters.py
# Related third party imports
import numpy as np

# Local application/library specific imports
from nnlibs.commons.io import padding


def convolution_compute_shapes(layer, A):
    """Compute forward shapes and dimensions for layer.
    """
    X = A    # Input of current layer

    X = padding(X, layer.d['p'])        #

    layer.fs['X'] = X.shape             # (m, ih, iw, n)

    layer.d['m'] = layer.fs['X'][0]     #
    layer.d['ih'] = layer.fs['X'][1]    #
    layer.d['iw'] = layer.fs['X'][2]    #
    layer.d['id'] = layer.fs['X'][3]    #

    # Apply to X_block - W shape is (filter_width, filter_width, image_depth, n_filters)
    layer.fs['W'] = (layer.d['fh'], layer.d['fw'], layer.d['id'], layer.d['n'])
    layer.fs['b'] = (1, 1, 1, layer.d['n'])

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

    #
    dZ = layer.bc['dZ']
    dZ = np.expand_dims(dZ, axis=3)
    dZ = np.expand_dims(dZ, axis=3)
    dZ = np.expand_dims(dZ, axis=3)

    # Gradients with respect to W
    dW = dZ * layer.fc['Xb']
    dW = np.sum(dW, axis=2)
    dW = np.sum(dW, axis=1)
    dW = np.sum(dW, axis=0)

    layer.g['dW'] = dW

    # Gradients with respect to b
    db = dW
    db = np.sum(db, axis=2, keepdims=True)
    db = np.sum(db, axis=1, keepdims=True)
    db = np.sum(db, axis=0, keepdims=True)

    layer.g['db'] = db if layer.use_bias else 0.

    return None


def convolution_update_parameters(layer):
    """Update parameters for layer.
    """
    for gradient in layer.g.keys():
        parameter = gradient[1:]
        # Update is driven by learning rate and gradients
        layer.p[parameter] -= layer.lrate[layer.e] * layer.g[gradient]

    return None
