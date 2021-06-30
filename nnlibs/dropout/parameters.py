#EpyNN/nnlibs/dropout/paremeters.py

import numpy as np

def init_mask(layer):

    layer.D = layer.np.random(layer.s['D'])

    layer.D = ( layer.D < layer.k )

    return None
