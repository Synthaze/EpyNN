# EpyNN/nnlibs/network/backward.py


def model_backward(model, dA):
    """Backward propagate error from output to input layer.
    """
    for layer in reversed(model.layers):

        dA = layer.backward(dA)

        layer.compute_gradients()
        layer.update_parameters()

    return None
