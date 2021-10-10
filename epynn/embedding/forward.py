# EpyNN/epynn/embedding/forward.py


def initialize_forward(layer, A):
    """Forward cache initialization.

    :param layer: An instance of embedding layer.
    :type layer: :class:`epynn.embedding.models.Embedding`

    :param A: Output of forward propagation from previous layer.
    :type A: :class:`numpy.ndarray`

    :return: Input of forward propagation for current layer.
    :rtype: :class:`numpy.ndarray`
    """
    X = layer.fc['X'] = A

    return X


def embedding_forward(layer, A):
    """Forward propagate signal to next layer.
    """
    # (1) Initialize cache
    X = initialize_forward(layer, A)

    # (2) Pass forward
    A = layer.fc['A'] = X

    return A   # To next layer
