#EpyNN/nnlibs/rnn/forward.py
import nnlibs.meta.parameters as mp

import nnlibs.rnn.parameters as rp

import numpy as np


def rnn_forward(layer,A):

    # Cache X (current) from A (prev)
    X = layer.fc['X'] = A
    layer.fs['X'] = X.shape

    # Init layer parameters
    if layer.init == True:
        rp.init_params(layer)

    # Init h shape
    layer.fs['h'] = ( layer.d['h'],layer.fs['X'][-1] )

    # Init h
    h = np.zeros(layer.fs['h'])

    # Loop through time steps
    for t in range(layer.fs['X'][1]):

        # Xt (X at current t) from X
        Xt = layer.fc['Xt'] = X[:,t]

        # Calculate input gate
        h = np.dot(layer.p['U'], Xt)
        h += np.dot(layer.p['V'], h) + layer.p['bh']
        h = layer.activate_input(h)

        # Calculate output gate
        A = np.dot( layer.p['W'], h ) + layer.p['bo']
        A = layer.activate_output(A)

        layer.fc['h'].append(h)
        layer.fc['A'].append(A)

    layer.fc = { k:np.array(v) for k,v in layer.fc.items() }

    A = layer.fc['A']

    return A
