#EpyNN/nnlibs/lstm/backward.py
import nnlibs.meta.parameters as mp

import numpy as np


def lstm_backward(layer,dA):

    mp.init_grads(layer)

    m = layer.s['X'][-1]

    dh_next = np.zeros(layer.s['h'])
    dC_next = np.zeros(layer.s['C'])

    for t in reversed(range(layer.s['X'][1])):

        dv = layer.c['v'][t] - dA[t]

        layer.g['dWv'] += 1./ m * np.dot(dv, layer.c['h'][t].T)
        layer.g['dbv'] += 1./ m * np.sum(dv,axis=1,keepdims=True)

        dh = np.dot(layer.p['Wv'].T, dv)

        dh += dh_next

        C_prev = layer.c['C'][t-1]

        do = dh * layer.c['c'][t]

        do = layer.derivative_output(do,layer.c['o'][t])

        layer.g['dWo'] += 1./ m * np.dot(do,layer.c['z'][t].T)
        layer.g['dbo'] += 1./ m * np.sum(do,axis=1,keepdims=True)

        dC = np.copy(dC_next)

        dC += layer.c['o'][t] * layer.derivative_memory(dh,layer.c['c'][t])

        dg = dC * layer.c['i'][t]

        dg = layer.derivative_candidate(dg,layer.c['g'][t])

        layer.g['dWg'] += 1./ m * np.dot(dg, layer.c['z'][t].T)
        layer.g['dbg'] += 1./ m * np.sum(dg,axis=1,keepdims=True)

        di = dC * layer.c['g'][t]

        di = layer.derivative_input(di,layer.c['i'][t])

        layer.g['dWi'] += 1./ m * np.dot(di, layer.c['z'][t].T)
        layer.g['dbi'] += 1./ m * np.sum(di,axis=1,keepdims=True)

        df = dC * C_prev

        df = layer.derivative(df,layer.c['f'][t])

        layer.g['dWf'] += 1./ m * np.dot(df, layer.c['z'][t].T)
        layer.g['dbf'] += 1./ m * np.sum(df,axis=1,keepdims=True)

        dz = (np.dot(layer.p['Wf'].T, df)
             + np.dot(layer.p['Wi'].T, di)
             + np.dot(layer.p['Wg'].T, dg)
             + np.dot(layer.p['Wo'].T, do))

        dh_prev = dz[:layer.d['h'], :]

        dC_prev = layer.c['f'][t] * dC

    mp.clip_gradient(layer)

    return None
