#EpyNN/nnlibs/rnn/forward.py
import nnlibs.rnn.parameters as rp

import numpy as np


def rnn_forward(layer,A):

    layer.X = A

    layer.h = []

    layer.A = []

    layer.s['X'] = layer.X.shape

    layer.s['h'] = ( layer.d['h'],layer.s['X'][-1] )

    h = np.zeros(layer.s['h'])

    if layer.init == True:
        rp.init_params(layer)

    for t in range(layer.s['X'][1]):

        h = np.dot(layer.p['U'], layer.X[:,t])

        h = h + np.dot(layer.p['V'], h)

        h = h + layer.p['bh']

        layer.h.append(layer.activate_input(h))

        layer.Z = np.dot(layer.p['W'], layer.h[-1]) + layer.p['bo']

        layer.A.append(layer.activate_output(layer.Z))

    layer.h = np.array(layer.h)
    layer.A = np.array(layer.A)

    return layer.A
