# EpyNN/epynn/embedding/backward.py


def initialize_backward(layer, dX):
    """Backward cache initialization.

    :param layer: An instance of embedding layer.
    :type layer: :class:`epynn.embedding.models.Embedding`

    :param dX: Output of backward propagation from next layer.
    :type dX: :class:`numpy.ndarray`

    :return: Input of backward propagation for current layer.
    :rtype: :class:`numpy.ndarray`
    """
    dA = layer.bc['dA'] = dX

    return dX


def embedding_backward(layer, dX):
    """Backward propagate error gradients to previous layer.
    """
    # (1) Initialize cache
    dA = initialize_backward(layer, dX)

    # (2) Pass backward
    dX = layer.bc['dX'] = dA

    return None    # No previous layer
