#EpyNN/nnlibs/meta/forward.py


def forward(model,A):

    for layer in model.l:

        A = layer.forward(A)

    return A
