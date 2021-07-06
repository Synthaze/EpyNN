#EpyNN/nnlibs/gru/forward.py
import nnlibs.gru.parameters as gp

import numpy as np


def gru_forward(layer,A):

    X, h = gp.init_forward(layer,A)

    # Init layer parameters
    if layer.init == True:
        gp.init_params(layer)

    # Loop through time steps
    for t in range(layer.ts):

        # Xt (X at current t) from X
        Xt = layer.fc['Xt'] = X[:,t]

        # Calculate update gate
        z = np.dot( layer.p['Wz'], Xt )
        z += np.dot( layer.p['Uz'], h ) + layer.p['bz']
        z = layer.fc['z'][t] = layer.activate_update(z)

        # Calculate reset gate
        r = np.dot( layer.p['Wr'], Xt )
        r += np.dot( layer.p['Ur'], h ) + layer.p['br']
        r = layer.fc['r'][t] = layer.activate_reset(r)

        # Calculate hidden units and input gate
        h = np.dot( layer.p['Wh'], Xt )
        h += np.dot(layer.p['Uh'], np.multiply(r,h) + layer.p['bh'])
        h = layer.fc['h'][t] = layer.activate_input(h)

        # Calculate hp units
        hp = layer.fc['hp'][t] = np.multiply(z,h) + np.multiply((1-z),h)

        # Calculate output gate
        At = np.dot(layer.p['Wy'], hp) + layer.p['by']
        At = layer.fc['A'][t] = layer.activate_output(At)

    A = gp.end_forward(layer)

    return A
