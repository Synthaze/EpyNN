#EpyNN/nnlibs/lstm/forward.py
import nnlibs.lstm.parameters as lp

import numpy as np


def lstm_forward(layer,A):

    X, hp, C = lp.init_forward(layer,A)

    # Init layer parameters
    if layer.init == True:
        lp.init_params(layer)

    # Loop through time steps
    for t in range(layer.ts):

        # Xt (X at current t) from X
        Xt = layer.fc['Xt'] = X[:,t]

        # Concatenate input and hidden state
        z = layer.fc['z'][t] = np.row_stack((hp, Xt))

        # Calculate forget gate
        f = np.dot(layer.p['Wf'], z) + layer.p['bf']
        f = layer.fc['f'][t] = layer.activate_forget(f)

        # Calculate input gate
        i = np.dot(layer.p['Wi'], z) + layer.p['bi']
        i = layer.fc['i'][t] = layer.activate_input(i)

        # Calculate candidate
        g = np.dot(layer.p['Wg'], z) + layer.p['bg']
        g = layer.fc['g'][t] = layer.activate_candidate(g)

        # Calculate output gate
        o = np.dot(layer.p['Wo'], z) + layer.p['bo']
        o = layer.fc['o'][t] = layer.activate_output(o)

        # Calculate memory state (prev)
        C = layer.fc['C'][t] = np.multiply(f,C) + np.multiply(i,g)

        # Calculate hidden state
        h = layer.fc['h'][t] = o * layer.activate_hidden(C)

        # Calculate logits
        At = np.dot(layer.p['Wv'], h) + layer.p['bv']
        At = layer.fc['A'][t] = layer.activate_logits(At)

    A = lp.end_forward(layer)

    return A
