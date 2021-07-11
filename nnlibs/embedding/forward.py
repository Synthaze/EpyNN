#EpyNN/nnlibs/embedding/forward.py
import nnlibs.embedding.parameters as ep

import numpy as np


def embedding_forward(layer,A):

    # If applicable - Init layer shapes and variables
    X = ep.init_forward(layer,A)

    # If applicable - Init layer parameters
    if layer.init == True:
        ep.init_params(layer)

    # Do stuff with X to compute A
    A = X

    return A
