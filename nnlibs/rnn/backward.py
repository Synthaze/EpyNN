#EpyNN/nnlibs/rnn/backward.py
import nnlibs.meta.parameters as mp

import numpy as np


def rnn_backward(layer,dA):

    mp.init_grads(layer)

    m = layer.s['X'][-1]

    d_h_next = np.zeros(layer.s['h'])

    for t in reversed(range(layer.s['X'][1])):

        d_o = layer.A[t] - dA[t]

        layer.g['dW'] += 1./ m * np.dot(d_o,layer.h[:,t].T)

        layer.g['dbo'] += 1./ m * np.sum(d_o,axis=1,keepdims=True)

        d_h = np.dot(layer.p['W'].T, d_o) + d_h_next

        d_f = layer.derivative_input(d_h,layer.h[:,t])

        layer.g['dbh'] += 1./ m * np.sum(d_f,axis=1,keepdims=True)

        layer.g['dU'] += 1./ m * np.dot(d_f, layer.X[:,t].T)

        layer.g['dV'] += 1./ m * np.dot(d_f, layer.h[t-1].T)

        d_h_next = np.dot(layer.p['V'].T, d_f)

    mp.clip_gradient(layer)

    return None
