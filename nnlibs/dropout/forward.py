# EpyNN/nnlibs/dropout/forward.py


def initialize_forward(layer, A):
    """Forward cache initialization.

    :param layer: An instance of dropout layer.
    :type layer: :class:`nnlibs.dropout.models.Dropout`

    :param A: Output of forward propagation from previous layer
    :type A: :class:`numpy.ndarray`

    :return: Input of forward propagation for current layer
    :rtype: :class:`numpy.ndarray`
    """
    X = layer.fc['X'] = A

    return X


def dropout_forward(layer, A):
    """Forward propagate signal to next layer.
    """
    # (1) Initialize cache
    X = initialize_forward(layer, A)

    # (2)
    D = layer.np_rng.uniform(0, 1, layer.fs['D'])

    # (3)
    D = layer.fc['D'] = (D < layer.d['k'])

    # (4)
    A = X * D
    A = A / layer.d['k']

    layer.fc['A'] = A

    return A    # To next layer
