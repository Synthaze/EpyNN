# EpyNN/nnlibs/embedding/forward.py


def initialize_forward(layer, A):
    """

    :param layer: An instance of the :class:`nnlibs.embedding.models.Embedding`
    :type layer: class:`nnlibs.embedding.models.Embedding`

    :param A: Output of forward propagation from previous layer
    :type A: class:`numpy.ndarray`

    :return: Input of forward propagation for current layer
    :rtype: class:`numpy.ndarray`
    """
    X = layer.fc['X'] = A

    return X


def embedding_forward(layer, A):
    """

    :param layer: An instance of the :class:`nnlibs.embedding.models.Embedding`
    :type layer: class:`nnlibs.embedding.models.Embedding`

    :param A: Output of forward propagation from previous layer
    :type A: class:`numpy.ndarray`

    :return: Output of forward propagation for current layer
    :rtype: class:`numpy.ndarray`
    """
    X = initialize_forward(layer, A)

    A = layer.fc['A'] = X.T

    return A   # To next layer
