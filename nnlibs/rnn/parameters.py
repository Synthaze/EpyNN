#EpyNN/nnlibs/rnn/parameters.py
import numpy as np


def init_params(layer):

    layer.p['U'] = np.random.randn(layer.s['U'][0],layer.s['U'][1])

    layer.p['V'] = np.random.randn(layer.s['V'][0],layer.s['V'][1])

    layer.p['W'] = np.random.randn(layer.s['W'][0],layer.s['W'][1])

    layer.p['bh'] = np.zeros(layer.s['bh'])

    layer.p['bo'] = np.zeros(layer.s['bo'])

    layer.init = False

    return None
