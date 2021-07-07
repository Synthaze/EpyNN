#EpyNN/nnlibs/meta/forward.py
import nnlibs.meta.parameters as mp

def forward(model,A):

    for layer in model.l:

        A = layer.forward(A)
        # Update shapes
        mp.update_shapes(layer)

    return A
