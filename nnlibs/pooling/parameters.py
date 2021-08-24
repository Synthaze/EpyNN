# EpyNN/nnlibs/pooling/parameters.py
# Local application/library specific imports
from nnlibs.commons.io import extract_blocks


def pooling_compute_shapes(layer, A):
    """Compute forward shapes and dimensions from input for layer.
    """
    X = A    # Input of current layer

    layer.fs['X'] = X.shape    # (m, h, w, d)

    layer.d['m'] = layer.fs['X'][0]    # Number of samples      (m)
    layer.d['h'] = layer.fs['X'][1]    # Height of features map (h)
    layer.d['w'] = layer.fs['X'][2]    # Width of features map  (w)
    layer.d['d'] = layer.fs['X'][3]    # Depth of features map  (d)

    return None


def pooling_initialize_parameters(layer):
    """Initialize parameters from shapes for layer.
    """
    # No parameters to initialize for Pooling layer

    return None


def pooling_compute_gradients(layer):
    """Compute gradients with respect to weight and bias for layer.
    """
    # No gradients to compute for Pooling layer

    return None


def pooling_update_parameters(layer):
    """Update parameters from gradients for layer.
    """
    # No parameters to update for Pooling layer

    return None
