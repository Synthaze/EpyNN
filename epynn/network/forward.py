# EpyNN/epynn/network/forward.py


def model_forward(model, X):
    """Forward propagate input data from input to output layer.
    """
    # By convention
    A = X

    # Iterate over layers
    for layer in model.layers:

        # For learning rate schedule
        layer.e = model.e

        # Layer returns A - layer.fs, layer.fc
        A = layer.forward(A)

    return A    # To derivative of loss function
