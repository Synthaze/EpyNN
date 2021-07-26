# EpyNN/nnlibs/meta/backward.py


def model_backward(model, dA):

    for layer in reversed(model.layers):

        dA = layer.backward(dA)

    return None
