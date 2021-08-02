# EpyNN/nnlibs/pool/parameters.py
# Standard library imports
import math

# Related third party imports
import numpy as np


def pooling_compute_shapes(layer, A):
    """Compute forward shapes and dimensions for layer.
    """
    X = A    # Input of current layer of shape (n .. m)

    layer.fs['X'] = X.shape

    dims = ['id', 'iw', 'ih','im']

    layer.d.update({d:i for d,i in zip(dims, layer.fs['X']})

    n_rows = layer.d['ih'] - layer.d['w'] + 1
    n_rows = min(layer.d['w'], n_rows)
    n_rows /= layer.d['s']

    n_cols = layer.d['ih'] - layer.d['w'] + 1
    # n_cols = min(layer.d['w'], n_cols)
    # n_cols /= layer.d['s']

    layer.d['R'] = math.ceil(n_rows)
    layer.d['C'] = math.ceil(n_cols)

    z_height = ((layer.d['ih'] - layer.d['w']) / layer.d['s']) + 1
    z_width = ((layer.d['iw'] - layer.d['w']) / layer.d['s']) + 1

    layer.d['zh'] = int(z_height)
    layer.d['zw'] = int(z_width)

    layer.fs['Z'] = (layer.d['im'], layer.d['zh'], layer.d['zw'], layer.d['id'])

    return None


def pooling_initialize_parameters(layer):
    """Initialize parameters for layer.
    """
    # No parameters to initialize for Pooling layer

    return None


def pooling_compute_gradients(layer):
    """Compute gradients with respect to weight and bias for layer.
    """
    # No gradients to update for Pooling layer

    return None


def pooling_update_parameters(layer):
    """Update parameters for layer.
    """
    # No parameters to update for Pooling layer

    return None
