# EpyNN/nnlibs/network/forward.py


def model_forward(model, X):
    """Forward propagate input data from input to output layer.
    """
    A = X

    for layer in model.layers:

        layer.e = model.e

        A = layer.forward(A)

    return A    # To derivative of ...
