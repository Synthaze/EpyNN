#EpyNN/nnlibs/dense/parameters.py
from nnlibs.commons.decorators import *

import numpy as np


@log_function
def init_params(layer):

    layer.s['X'] = layer.X.shape

    layer.s['W'] = ( layer.d['d'], layer.s['X'][0] )

    layer.s['b'] = ( layer.d['d'], 1 )

    layer.p['W'] = np.random.randn(layer.s['W'][0],layer.s['W'][1]) / np.sqrt(layer.s['W'][1])

    layer.p['b'] = np.zeros(layer.s['b'])

    layer.init = False

    return None
