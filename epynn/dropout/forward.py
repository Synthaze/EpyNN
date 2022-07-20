# EpyNN/epynn/dropout/forward.py
# Related third party imports
import numpy as np


def initialize_forward(layer, A):
    """Forward cache initialization.

    :param layer: An instance of dropout layer.
    :type layer: :class:`epynn.dropout.models.Dropout`

    :param A: Output of forward propagation from previous layer.
    :type A: :class:`numpy.ndarray`

    :return: Input of forward propagation for current layer.
    :rtype: :class:`numpy.ndarray`
    """
    X = layer.fc['X'] = A

    return X


def dropout_forward(layer, A):
    """Forward propagate signal to next layer.
    """
    # (1) Initialize cache
    X = initialize_forward(layer, A)

    # (2) Generate dropout mask
    D = layer.np_rng.uniform(0, 1, layer.fs['D'])

    # (3) Apply a step function with respect to drop_prob (k)
    D = layer.fc['D'] = (D > layer.d['d'])

    # (4) Drop data points
    A = X * D

    # (5) Scale up signal
    A /= (1 - layer.d['d'])

    return A    # To next layer
