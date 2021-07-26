#EpyNN/nnlibs/gru/forward.py
import nnlibs.gru.parameters as gp

import numpy as np


def gru_forward(layer,A):

    X, hp = gp.init_forward(layer,A)

    # Init layer parameters
    if layer.init == True:
        gp.init_params(layer)

    # Loop through time steps
    for t in range(layer.ts):

        # Xt (X at current t) from X
        Xt = layer.fc['Xt'] = X[:,t]

        # Calculate reset gate
        r = np.dot( layer.p['Wr'], Xt )
        r += np.dot( layer.p['Ur'], hp ) + layer.p['br']
        r = layer.fc['r'][t] = layer.activate_reset(r)

        # Calculate update gate
        z = np.dot( layer.p['Wz'], Xt )
        z += np.dot( layer.p['Uz'], hp ) + layer.p['bz']
        z = layer.fc['z'][t] = layer.activate_update(z)

        # Calculate hidden units and input gate
        hh = np.dot( layer.p['Wh'], Xt )
        hh += np.dot( layer.p['Uh'], np.multiply( r, hp ) ) + layer.p['bh']
        hh = layer.fc['hh'][t] = layer.activate_input(hh)

        # Calculate hp units
        h = hp = layer.fc['h'][t] = np.multiply(z,hp) + np.multiply((1-z),hh)

        # Calculate output gate
        At = np.dot(layer.p['Wy'], h) + layer.p['by']
        At = layer.fc['A'][t] = layer.activate_output(At)

    A = gp.end_forward(layer)

    return A
