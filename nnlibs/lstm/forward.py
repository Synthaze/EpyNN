#EpyNN/nnlibs/lstm/forward.py
import nnlibs.lstm.parameters as lp

import numpy as np


def lstm_forward(layer,A):

    layer.X = A

    layer.init_cache()

    layer.A = []

    layer.s['X'] = layer.X.shape

    layer.s['h'] = ( layer.d['h'],layer.s['X'][-1] )
    layer.s['C'] = ( layer.d['h'],layer.s['X'][-1] )

    h = np.zeros(layer.s['h'])
    C = np.zeros(layer.s['C'])

    if layer.init == True:
        lp.init_params(layer)

    for t in range(layer.s['X'][1]):

        # Concatenate input and hidden state
        z = np.row_stack((h, layer.X[:,t]))

        # Calculate forget gate
        f = np.dot(layer.p['Wf'], z) + layer.p['bf']
        f = layer.activate_forget(f)

        # Calculate input gate
        i = np.dot(layer.p['Wi'], z) + layer.p['bi']
        i = layer.activate_input(i)

        # Calculate candidate
        g = np.dot(layer.p['Wg'], z) + layer.p['bg']
        g = layer.activate_candidate(g)

        # Calculate memory state
        C = f * C + i * g
        c = layer.activate_memory(C)

        # Calculate output gate
        o = np.dot(layer.p['Wo'], z) + layer.p['bo']
        o = layer.activate_output(o)

        # Calculate hidden state
        h = o * layer.activate_input(C)

        # Calculate logits
        v = np.dot(layer.p['Wv'], h) + layer.p['bv']

        # Calculate softmax
        v = layer.activate_output(v)

        layer.c['h'].append(h)
        layer.c['C'].append(C)
        layer.c['z'].append(z)
        layer.c['f'].append(f)
        layer.c['i'].append(i)
        layer.c['g'].append(g)
        layer.c['z'].append(z)
        layer.c['o'].append(o)
        layer.c['v'].append(v)

        layer.A.append(v)

    layer.array_cache()

    layer.A = np.array(layer.A)

    return layer.A
