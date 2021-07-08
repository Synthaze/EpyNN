#EpyNN/nnlibs/rnn/backward.py
import nnlibs.meta.parameters as mp

import nnlibs.rnn.parameters as rp

import numpy as np


def rnn_backward(layer,dA):

    dX, dhn, dXt, dh = rp.init_backward(layer,dA)

    # Loop through time steps
    for t in reversed(range(layer.ts)):

        if layer.binary == False:
            # Cache dXt (dX at current t) from dX
            dXt = layer.bc['dXt'] = dX[t]
            # Cache dh (current) from dXt (prev), dhn, z (current) and h (current)
            dh = layer.bc['dh'] = np.dot(layer.p['W'].T, dXt) + dhn

        # Cache dh (current) from dXt (prev), dhn, z (current) and h (current)
        df = layer.bc['df'] = layer.derivative_output(dh,layer.fc['h'][t])

        # Update gradients
        rp.update_gradients(layer,t)

        # Cache dhn
        dhn = layer.bc['dhn'] = np.dot(layer.p['V'].T, df)

    # Clip gradients
    mp.clip_gradient(layer)

    return None
