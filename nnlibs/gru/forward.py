#EpyNN/nnlibs/gru/forward.py
import nnlibs.gru.parameters as gp

import numpy as np


def gru_forward(layer,A):

    # Cache X (current) from A (prev)
    X = layer.fc['X'] = A
    layer.fs['X'] = X.shape

    # Init layer parameters
    if layer.init == True:
        gp.init_params(layer)

    # Init h shape
    layer.fs['h'] = ( layer.d['h'],layer.fs['X'][-1] )

    # Init h
    h = np.zeros(layer.fs['h'])

    # Loop through time steps
    for t in range(layer.fs['X'][1]):

        # Xt (X at current t) from X
        Xt = layer.fc['Xt'] = X[:,t]

        # Calculate update gate
        z = np.dot( layer.p['Wz'], Xt )
        z += np.dot( layer.p['Uz'], h ) + layer.p['bz']
        z = layer.activate_update(z)
        
        # Calculate reset gate
        r = np.dot( layer.p['Wr'], Xt )
        r += np.dot( layer.p['Ur'], h ) + layer.p['br']
        r = layer.activate_reset(r)

        # Calculate hidden units and input gate
        h = np.dot( layer.p['Wh'], Xt )
        h += np.dot(layer.p['Uh'], np.multiply(r,h) + layer.p['bh'])
        h = layer.activate_input(h)

        # Calculate hp units
        hp = np.multiply(z,h) + np.multiply((1-z),h)

        # Calculate output gate
        A = np.dot(layer.p['Wy'], hp) + layer.p['by']
        A = layer.activate_output(A)

        layer.fc['z'].append(z)
        layer.fc['r'].append(r)
        layer.fc['h'].append(h)
        layer.fc['hp'].append(hp)
        layer.fc['A'].append(A)

    layer.fc = { k:np.array(v) for k,v in layer.fc.items() }

    return layer.fc['A']
