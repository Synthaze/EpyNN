# EpyNN/nnlibs/flatten/forward.py


def initialize_forward(layer, A):
    """Forward cache initialization.

    :param layer: An instance of the :class:`nnlibs.flatten.models.Flatten`
    :type layer: class:`nnlibs.flatten.models.Flatten`

    :param A: Output of forward propagation from previous layer
    :type A: class:`numpy.ndarray`

    :return: Input of forward propagation for current layer
    :rtype: class:`numpy.ndarray`
    """
    X = layer.fc['X'] = A

    return X


def flatten_forward(layer,A):
    """Forward propagate signal to next layer.
    """
    # (1) Initialize cache
    X = initialize_forward(layer, A)

    # (2) Reshape (m, s, v) to (m, sv)
    A = layer.fc['A'] = X.reshape(layer.fs['A'])

    return A   # To next layer
