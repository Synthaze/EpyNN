# EpyNN/nnlibs/dropout/forward.py


def initialize_forward(layer, A):
    """Forward cache initialization.

    :param layer: An instance of dropout layer.
    :type layer: :class:`nnlibs.dropout.models.Dropout`

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

    # (3) Apply a step function with respect to keep_prob (k)
    D = layer.fc['D'] = (D > layer.d['d'])

    # (4) Drop data points
    A = X * D

    # (5) Scale up signal
    A /= layer.d['s']

    return A    # To next layer
