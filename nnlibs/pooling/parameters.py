# EpyNN/nnlibs/pooling/parameters.py
# Standard library imports
import math

# Related third party imports
import numpy as np


def pooling_compute_shapes(layer, A):
    """Compute forward shapes and dimensions for layer.
    """
    X = A    # Input of current layer of shape (m .. n)

    layer.fs['X'] = X.shape

    dims = ['m', 'ih', 'iw', 'id']

    layer.d.update({d:i for d,i in zip(dims, layer.fs['X'])})

    layer.d['oh'] = math.ceil(min(layer.d['w'], layer.d['ih'] - layer.d['h'] + 1) / layer.d['s'])
    layer.d['ow'] = math.ceil(min(layer.d['w'], layer.d['iw'] - layer.d['w'] + 1) / layer.d['s'])

    layer.d['zh'] = int(((layer.d['ih'] - layer.d['w']) / layer.d['s']) + 1)
    layer.d['zw'] = int(((layer.d['iw'] - layer.d['w']) / layer.d['s']) + 1)

    layer.fs['Z'] = (layer.d['m'], layer.d['zh'], layer.d['zw'], layer.d['id'])

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
