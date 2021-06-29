#EpyNN/nnlibs/conv/parameters.py
from nnlibs.commons.decorators import *

import numpy as np


@log_function
def init_params(layer):

    layer.p['W'] = np.random.random( layer.s['W'] ) * 0.1
    layer.p['b'] = np.random.random( layer.s['b'] ) * 0.01

    layer.init = False

    return None
