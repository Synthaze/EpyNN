#EpyNN/nnlibs/template/forward.py
import nnlibs.meta.parameters as mp

import nnlibs.template.parameters as tp

import numpy as np


def template_forward(layer,A):

    # If applicable - Init layer shapes and variables
    X = tp.init_forward(layer,A)

    # If applicable - Init layer parameters
    if layer.init == True:
        tp.init_params(layer)

    # Do stuff with X to compute A
    A = X

    return A
