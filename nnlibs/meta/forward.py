# EpyNN/nnlibs/meta/forward.py


def model_forward(model, A):

    for layer in model.layers:

        A = layer.forward(A)

    return A
