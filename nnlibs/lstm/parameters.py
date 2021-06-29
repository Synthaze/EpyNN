#EpyNN/nnlibs/lstm/parameters.py
import numpy as np


def init_params(layer):

    layer.p['Wf'] = np.random.randn(layer.s['Wf'][0], layer.s['Wf'][1])
    layer.p['Wi'] = np.random.randn(layer.s['Wi'][0], layer.s['Wi'][1])
    layer.p['Wg'] = np.random.randn(layer.s['Wg'][0], layer.s['Wg'][1])
    layer.p['Wv'] = np.random.randn(layer.s['Wv'][0], layer.s['Wv'][1])
    layer.p['Wo'] = np.random.randn(layer.s['Wo'][0], layer.s['Wo'][1])

    layer.p['bf'] = np.zeros(layer.s['bf'])
    layer.p['bi'] = np.zeros(layer.s['bi'])
    layer.p['bg'] = np.zeros(layer.s['bg'])
    layer.p['bv'] = np.zeros(layer.s['bv'])
    layer.p['bo'] = np.zeros(layer.s['bo'])

    layer.init = False

    return None
