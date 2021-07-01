#EpyNN/nnlibs/meta/backward.py
import nnlibs.meta.parameters as mp


def backward(model,dA):

    for layer in reversed(model.l):

        mp.init_grads(layer)

        dA = layer.backward(dA)
        # Update shapes
        mp.update_shapes(layer)

    return None
