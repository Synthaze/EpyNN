#EpyNN/nnlibs/lstm/forward.py
import nnlibs.meta.parameters as mp

import nnlibs.lstm.parameters as lp

import numpy as np


def lstm_forward(layer,A):

    # Cache X (current) from A (prev)
    X = layer.fc['X'] = A
    layer.fs['X'] = X.shape

    # Init layer parameters
    if layer.init == True:
        lp.init_params(layer)

    # Init h and C shapes
    layer.fs['h'] = ( layer.d['h'],layer.fs['X'][-1] )
    layer.fs['C'] = ( layer.d['h'],layer.fs['X'][-1] )

    # Init h and C
    h = np.zeros(layer.fs['h'])
    C = np.zeros(layer.fs['C'])

    # Loop through time steps
    for t in range(layer.fs['X'][1]):

        # Xt (X at current t) from X
        Xt = layer.fc['Xt'] = X[:,t]

        # Concatenate input and hidden state
        z = np.row_stack((h, Xt))

        # Calculate forget gate
        f = np.dot(layer.p['Wf'], z) + layer.p['bf']
        f = layer.activate_candidate(f)

        # Calculate input gate
        i = np.dot(layer.p['Wi'], z) + layer.p['bi']
        i = layer.activate_candidate(i)

        # Calculate candidate
        g = np.dot(layer.p['Wg'], z) + layer.p['bg']
        g = layer.activate_candidate(g)

        # Calculate output gate
        o = np.dot(layer.p['Wo'], z) + layer.p['bo']
        o = layer.activate_candidate(o)

        # Calculate memory state (prev)
        C = np.multiply(f,C) + np.multiply(i,g)

        # Calculate hidden state
        h = o * layer.activate_input(C)

        # Calculate logits
        A = np.dot(layer.p['Wv'], h) + layer.p['bv']
        A = layer.activate_output(A)

        # Memory state c (current)
        c = layer.activate_memory(C)

        layer.fc['h'].append(h)
        layer.fc['C'].append(C)
        layer.fc['c'].append(C)
        layer.fc['z'].append(z)
        layer.fc['f'].append(f)
        layer.fc['i'].append(i)
        layer.fc['g'].append(g)
        layer.fc['z'].append(z)
        layer.fc['o'].append(o)
        layer.fc['A'].append(A)

    layer.fc = { k:np.array(v) for k,v in layer.fc.items() }

    A = layer.fc['A']

    return A
