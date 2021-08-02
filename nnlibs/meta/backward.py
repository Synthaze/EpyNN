# EpyNN/nnlibs/meta/backward.py


def model_backward(model, dA):
    """Backward propagate error from output to input layer.

    :param model: An instance of EpyNN network.
    :type model: :class:`nnlibs.meta.models.EpyNN`

    :param dA: Derivative of cost function with respect to output of forward propagation
    :type dA: :class:`numpy.ndarray`
    """
    for layer in reversed(model.layers):

        dA = layer.backward(dA)

        layer.compute_gradients()
        layer.update_parameters()

    return None
