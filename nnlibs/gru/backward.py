#EpyNN/nnlibs/gru/backward.py
import nnlibs.meta.parameters as mp

import numpy as np


def gru_backward(layer,dA):

    mp.init_grads(layer)

    m = layer.s['X'][-1]

    dh_next = np.zeros(layer.s['h'])

    for t in reversed(range(layer.s['X'][1])):

        d_y = layer.A[t].copy()

        d_y = d_y - dA[t]

        layer.g['dWy'] += 1./ m * np.dot(d_y, layer.c['h'][t].T)

        layer.g['dby'] += 1./ m * np.sum(d_y,axis=1,keepdims=True)

        d_h = np.dot(layer.p['Wy'].T, d_y) + dh_next

        d_h = np.multiply(d_h,(1-layer.c['z'][t]))

        d_h = layer.derivative_input(d_h,layer.c['h'][t])

        layer.g['dWh'] += 1./ m * np.dot(d_h, layer.X[:,t].T)

        layer.g['dUh'] += 1./ m * np.dot(d_h, np.multiply(layer.c['r'][t], layer.c['h'][t-1]).T)

        layer.g['dbh'] += 1./ m * np.sum(d_h,axis=1,keepdims=True)

        d_r = np.dot(layer.p['Uh'].T, d_h)
        d_r = np.multiply(d_r, layer.c['h'][t-1])

        d_r = layer.derivative_reset(d_r,layer.c['r'][t])

        layer.g['dWr'] += 1./ m * np.dot(d_r, layer.X[:,t].T)

        layer.g['dUr'] += 1./ m * np.dot(d_r, layer.c['h'][t-1].T)

        layer.g['dbr'] += 1./ m * np.sum(d_r,axis=1,keepdims=True)

        d_z = np.multiply(d_h, layer.c['h'][t-1] - layer.c['h'][t])
        d_z = layer.derivative_update(d_z,layer.c['z'][t])

        layer.g['dWz'] += 1./ m * np.dot(d_z, layer.X[:,t].T)

        layer.g['dUz'] += 1./ m * np.dot(d_z, layer.c['h'][t-1].T)

        layer.g['dbz'] += 1./ m * np.sum(d_z,axis=1,keepdims=True)

        # All influences of previous layer to loss
        d_h_fz_i = np.dot(layer.p['Uz'].T, d_z)

        d_h_fz = np.multiply(d_h, layer.c['z'][t])

        d_h_fhh = np.multiply(np.dot(layer.p['Uh'].T, d_h), layer.c['r'][t])

        d_h_fr = np.dot(layer.p['Ur'].T, d_r)

        # ∂loss/∂h𝑡₋₁
        dh_next = d_h_fz_i + d_h_fz + d_h_fhh + d_h_fr


    mp.clip_gradient(layer)

    return None
