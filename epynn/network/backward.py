# EpyNN/epynn/network/backward.py


def model_backward(model, dA):
    """Backward propagate error gradients from output to input layer.
    """
    # By convention
    dX = dA

    # Iterate over reversed layers
    for layer in reversed(model.layers):

        # Layer returns dL/dX (dX) - layer.bs, layer.bc
        dX = layer.backward(dX)

        # Update values in layer.g
        layer.compute_gradients()

        # Update values in layer.p
        layer.update_parameters()

    return None
