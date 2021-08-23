# EpyNN/nnlibs/network/backward.py


def model_backward(model, dA):
    """Backward propagate error from output to input layer.
    """
    #
    dX = dA

    for layer in reversed(model.layers):

        dX = layer.backward(dX)

        layer.compute_gradients()
        layer.update_parameters()

    return dX
