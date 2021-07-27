# EpyNN/nnlibs/embedding/parameters.py
# Related third party imports
import numpy as np


def embedding_compute_shapes(layer, A):
    """Compute shapes for Embedding layer object

    :param layer: An instance of the :class:`nnlibs.embedding.models.Embedding`
    :type layer: class:`nnlibs.embedding.models.Embedding`
    """

    X = A

    layer.fs['X'] = X.shape

    layer.d['m'] = layer.fs['X'][0]
    layer.d['n'] = layer.fs['X'][1]

    return None


def embedding_initialize_parameters(layer):
    """Dummy function - Initialize parameters for Embedding layer object

    :param layer: An instance of the :class:`nnlibs.embedding.models.Embedding`
    :type layer: class:`nnlibs.embedding.models.Embedding`
    """

    # No parameters to initialize for Embedding layer

    return None


def embedding_compute_gradients(layer):
    """Dummy function - Update weight and bias gradients for Embedding layer object

    :param layer: An instance of the :class:`nnlibs.embedding.models.Embedding`
    :type layer: class:`nnlibs.embedding.models.Embedding`
    """

    # No gradients to update for Embedding layer

    return None


def embedding_update_parameters(layer):
    """Dummy function - Update parameters for Embedding layer object

    :param layer: An instance of the :class:`nnlibs.embedding.models.Embedding`
    :type layer: class:`nnlibs.embedding.models.Embedding`
    """

    # No parameters to update for Embedding layer

    return None
