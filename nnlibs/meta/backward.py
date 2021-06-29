#EpyNN/nnlibs/meta/backward.py


def backward(model,dA):

    for layer in reversed(model.l):

        dA = layer.backward(dA)

    return None
