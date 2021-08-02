# EpyNN/nnlibs/meta/forward.py


def model_forward(model, X):
    """Forward propagate input data from input to output layer.

    :param model: An instance of EpyNN network.
    :type model: :class:`nnlibs.meta.models.EpyNN`

    :param X: Input of forward propagation for embedding layer
    :type X: :class:`numpy.ndarray`

    :return: Output of forward propagation for output layer
    :rtype: :class:`numpy.ndarray`
    """
    A = X

    for layer in model.layers:

        layer.e = model.e

        A = layer.forward(A)

    return A
