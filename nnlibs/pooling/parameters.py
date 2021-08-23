# EpyNN/nnlibs/pooling/parameters.py
# Standard library imports
import math

# Related third party imports
import numpy as np


def pooling_compute_shapes(layer, A):
    """Compute forward shapes and dimensions for layer.
    """
    X = A    # Input of current layer of shape (m, ih, iw, n)

    layer.fs['X'] = X.shape    # (m, ih, iw, n)

    dims = ['m', 'ih', 'iw', 'n']

    layer.d.update({d: i for d, i in zip(dims, layer.fs['X'])})

    layer.d['oh'] = math.ceil(min(layer.d['ph'], layer.d['ih'] - layer.d['ph'] + 1) / layer.d['sh'])
    layer.d['ow'] = math.ceil(min(layer.d['pw'], layer.d['iw'] - layer.d['pw'] + 1) / layer.d['sw'])

    layer.d['zh'] = math.floor(((layer.d['ih'] - layer.d['ph']) / layer.d['sh'])) + 1
    layer.d['zw'] = math.floor(((layer.d['iw'] - layer.d['pw']) / layer.d['sw'])) + 1

    layer.fs['Z'] = (layer.d['m'], layer.d['zh'], layer.d['zw'], layer.d['n'])



    return None


def pooling_initialize_parameters(layer):
    """Initialize parameters for layer.
    """
    # No parameters to initialize for Pooling layer

    return None


def pooling_compute_gradients(layer):
    """Compute gradients with respect to weight and bias for layer.
    """
    # No gradients to compute for Pooling layer

    return None


def pooling_update_parameters(layer):
    """Update parameters for layer.
    """
    # No parameters to update for Pooling layer

    return None
