#EpyNN/nnlibs/gru/forward.py
import nnlibs.gru.parameters as gp

import numpy as np


def gru_forward(layer,A):

    layer.X = A

    layer.init_cache()

    layer.A = []

    layer.s['X'] = layer.X.shape

    layer.s['h'] = ( layer.d['h'],layer.s['X'][-1] )

    h = np.zeros(layer.s['h'])

    if layer.init == True:
        gp.init_params(layer)

    for t in range(layer.s['X'][1]):

        # Calculate update and reset gates
        z = np.dot(layer.p['Wz'],layer.X[:,t]) + np.dot(layer.p['Uz'],h) + layer.p['bz']
        z = layer.activate_update(z)

        r = np.dot(layer.p['Wr'],layer.X[:,t]) + np.dot(layer.p['Ur'],h) + layer.p['br']
        r = layer.activate_reset(r)

        # Calculate hidden units
        h = np.dot(layer.p['Wh'],layer.X[:,t]) + np.dot(layer.p['Uh'], np.multiply(r,h) + layer.p['bh'])
        h = layer.activate_input(h)

        h_prev = np.multiply(z,h) + np.multiply((1-z),h)

        A = np.dot(layer.p['Wy'], h_prev) + layer.p['by']
        A = layer.activate_output(A)

        layer.c['z'].append(z)
        layer.c['r'].append(r)
        layer.c['h'].append(h)
        layer.c['hp'].append(h_prev)

        layer.A.append(A)

    layer.array_cache()

    layer.A = np.array(layer.A)

    return layer.A
