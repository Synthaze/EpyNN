# EpyNN/nnlibs/dropout/paremeters.py
# Related third party imports
import numpy as np


def dropout_compute_shapes(layer, A):
    """Compute shapes for Dropout layer object

    :param layer: An instance of the :class:`nnlibs.dropout.models.Dropout`
    :type layer: class:`nnlibs.dropout.models.Dropout`
    """

    X = A

    layer.fs['X'] = layer.fs['D'] = X.shape

    layer.d['m'] = layer.fs['X'][-1]
    layer.d['n'] = X.size // layer.d['m']

    return None


def dropout_initialize_parameters(layer):
    """Dummy function - Initialize parameters for Dropout layer object

    :param layer: An instance of the :class:`nnlibs.dropout.models.Dropout`
    :type layer: class:`nnlibs.dropout.models.Dropout`
    """

    # No parameters to initialize for Dropout layer

    return None


def dropout_update_gradients(layer):
    """Dummy function - Update weight and bias gradients for Dropout layer object

    :param layer: An instance of the :class:`nnlibs.dropout.models.Dropout`
    :type layer: class:`nnlibs.dropout.models.Dropout`
    """

    # No gradients to update for Dropout layer

    return None


def dropout_update_parameters(layer):
    """Dummy function - Update parameters for Dropout layer object

    :param layer: An instance of the :class:`nnlibs.dropout.models.Dropout`
    :type layer: class:`nnlibs.dropout.models.Dropout`
    """

    # No parameters to update for Dropout layer

    return None
