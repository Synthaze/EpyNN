# EpyNN/nnlibs/meta/parameters.py
# Related third party imports
import numpy as np

# Local application/library specific imports
from nnlibs.commons.schedule import schedulers


def assign_seed_layers(model):

    seed = model.seed

    for layer in model.layers:

        if seed != None:
            layer.o['seed'] = seed
            seed += 1

        else:
            layer.o['seed'] = None

        layer.np_rng = np.random.default_rng(seed=layer.o['seed'])

    return None


def compute_learning_rate(model):

    for layer in model.layers:

        if layer.se_hPars == None:
            se_hPars = layer.se_hPars = model.se_hPars
        else:
            se_hPars = layer.se_hPars

        layer.se_hPars, layer.lrate = schedulers(se_hPars)

    return None

#
# def clip_gradient(layer,max_norm=0.25):
#
#     # Set the maximum of the norm to be of type float
#     max_norm = float(max_norm)
#     total_norm = 0
#
#     # Calculate the L2 norm squared for each gradient and add them to the total norm
#     for grad in layer.g.values():
#         grad_norm = np.sum(np.power(grad, 2))
#         total_norm += grad_norm
#
#     total_norm = np.sqrt(total_norm)
#
#     # Calculate clipping coeficient
#     clip_coef = max_norm / (total_norm + 1e-6)
#
#     # If the total norm is larger than the maximum allowable norm, then clip the gradient
#     if clip_coef < 1:
#         for g in layer.g.keys():
#             layer.g[g] *= clip_coef
#
#     return None
