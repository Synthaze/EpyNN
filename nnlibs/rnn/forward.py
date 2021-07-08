#EpyNN/nnlibs/rnn/forward.py
import nnlibs.rnn.parameters as rp

import numpy as np


def rnn_forward(layer,A):

    X, hp = rp.init_forward(layer,A)

    # Init layer parameters
    if layer.init == True:
        rp.init_params(layer)

    # Loop through time steps
    for t in range(layer.ts):

        # Xt (X at current t) from X
        Xt = layer.fc['Xt'] = X[:,t]

        # Calculate input gate
        h = np.dot(layer.p['U'], Xt)
        h += np.dot(layer.p['V'], hp) + layer.p['bh']
        h = hp = layer.fc['h'][t] = layer.activate_input(h)

        # Calculate output gate
        At = np.dot( layer.p['W'], h ) + layer.p['bo']
        At = layer.fc['A'][t] = layer.activate_output(At)

    A = rp.end_forward(layer)

    return A
