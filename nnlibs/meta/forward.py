# EpyNN/nnlibs/meta/forward.py


def model_forward(model, A):
    """.

    :param A:
    :type A:
    :return:
    :rtype:
    """
    for layer in model.layers:

        layer.e = model.e

        A = layer.forward(A)

    return A
