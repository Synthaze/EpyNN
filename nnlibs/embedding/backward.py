#EpyNN/nnlibs/embedding/backward.py
import nnlibs.embedding.parameters as ep

import numpy as np


def embedding_backward(layer,dA):

    # If applicable - Init layer shapes and variables
    dX = ep.init_backward(layer,dA)

    # Do stuff with dX to compute dA
    dA = dX

    return dA
