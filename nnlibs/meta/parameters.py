#EpyNN/nnlibs/meta/parameters.py
import numpy as np


def update_params(model,hPars):

    for layer in model.l:

        for gradient in layer.g.keys():

            parameter = gradient[1:]

            layer.p[parameter] -= hPars.l[hPars.e] * layer.g[gradient]

    return None


def init_grads(layer):

    for parameter in layer.p.keys():

        gradient = 'd'+parameter

        layer.g[gradient] = np.zeros_like(layer.p[parameter])

    return None


def clip_gradient(layer,max_norm=0.25):

    # Set the maximum of the norm to be of type float
    max_norm = float(max_norm)
    total_norm = 0

    # Calculate the L2 norm squared for each gradient and add them to the total norm
    for grad in layer.g.values():
        grad_norm = np.sum(np.power(grad, 2))
        total_norm += grad_norm

    total_norm = np.sqrt(total_norm)

    # Calculate clipping coeficient
    clip_coef = max_norm / (total_norm + 1e-6)

    # If the total norm is larger than the maximum allowable norm, then clip the gradient
    if clip_coef < 1:
        for g in layer.g.keys():
            layer.g[g] *= clip_coef

    return None
