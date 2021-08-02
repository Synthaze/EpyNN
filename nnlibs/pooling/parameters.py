# EpyNN/nnlibs/pool/parameters.py
# Standard library imports
import math

# Related third party imports
import numpy as np


def pooling_compute_shapes(layer, A):
    """Compute forward shapes and dimensions for layer.
    """
    X = A    # Input of current layer of shape (m .. n)

    layer.fs['X'] = X.shape

    dims = ['m', 'ih', 'iw', 'n']

    layer.d.update({d:i for d,i in zip(dims, layer.fs['X'])})

    oh = layer.d['oh'] = math.floor((layer.d['ih'] - layer.d['w']) / layer.d['s']) + 1
    ow = layer.d['ow'] = math.floor((layer.d['iw'] - layer.d['w']) / layer.d['s']) + 1

    layer.fs['Z'] = (layer.d['m'], layer.d['oh'], layer.d['ow'], layer.d['n'])

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
