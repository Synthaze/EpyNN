# EpyNN/nnlibs/pool/parameters.py
# Standard library imports
import math

# Related third party imports
import numpy as np


def pooling_compute_shapes(layer, A):
    """Compute shapes for Pooling layer object

    :param layer: An instance of the :class:`nnlibs.pooling.models.Pooling`
    :type layer: class:`nnlibs.pooling.models.Pooling`
    """

    X = A

    layer.fs['X'] = X.shape

    layer.d['id'], layer.d['iw'], layer.d['ih'], layer.d['im'] = layer.fs['X']

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

    layer.fs['Z'] = (layer.d['im'], layer.d['zh'], layer.d['zw'], layer.d['id'])

    return None


def pooling_initialize_parameters(layer):
    """Dummy function - Initialize parameters for Pooling layer object

    :param layer: An instance of the :class:`nnlibs.pooling.models.Pooling`
    :type layer: class:`nnlibs.pooling.models.Pooling`
    """

    # No parameters to initialize for Pooling layer

    return None


def pooling_update_gradients(layer):
    """Dummy function - Update weight and bias gradients for Pooling layer object

    :param layer: An instance of the :class:`nnlibs.pooling.models.Pooling`
    :type layer: class:`nnlibs.pooling.models.Pooling`
    """

    # No gradients to update for Pooling layer

    return None


def pooling_update_parameters(layer):
    """Dummy function - Update parameters for Pooling layer object

    :param layer: An instance of the :class:`nnlibs.pooling.models.Pooling`
    :type layer: class:`nnlibs.pooling.models.Pooling`
    """

    # No parameters to update for Pooling layer

    return None
