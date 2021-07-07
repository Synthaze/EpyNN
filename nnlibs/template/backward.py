#EpyNN/nnlibs/template/backward.py
import nnlibs.meta.parameters as mp

import nnlibs.template.parameters as tp

import numpy as np


def template_backward(layer,dA):

    # If applicable - Init layer shapes and variables
    dX = tp.init_backward(layer,dA)

    # Do stuff with dX to compute dA
    dA = dX

    # If applicable - Update layer gradients
    tp.update_grads(layer)

    return dA
