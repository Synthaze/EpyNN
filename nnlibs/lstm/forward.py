# EpyNN/nnlibs/lstm/forward.py
# Related third party imports
import numpy as np


def lstm_forward(layer, A):

    X, hp, C = initialize_forward(layer, A)

    # Loop through time steps
    for t in range(layer.d['t']):

        # Xt (X at current t) from X
        X = layer.fc['Xt'][t] = layer.fc['X'][:, t]

        # Concatenate input and hidden state
        z = layer.fc['z'][t] = np.row_stack((hp, X))

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
        C = layer.fc['C'][t] = f*C + i*g

        # Calculate hidden state
        h = layer.fc['h'][t] = o * layer.activate_hidden(C)

        # Calculate logits
        A = np.dot(layer.p['W'], h) + layer.p['b']
        A = layer.fc['A'][t] = layer.activate(A)

    if layer.binary == True:
        A = layer.fc['A'] = layer.fc['A'][-1]
    else:
        A = layer.fc['A']

    return A


def initialize_forward(layer, A):

    X = layer.fc['X'] = A

    layer.fc['Xt'] = np.zeros(layer.fs['Xt'])
    layer.fc['h'] = np.zeros(layer.fs['h'])
    layer.fc['o'] = np.zeros_like(layer.fc['h'])
    layer.fc['i'] = np.zeros_like(layer.fc['h'])
    layer.fc['f'] = np.zeros_like(layer.fc['h'])
    layer.fc['g'] = np.zeros_like(layer.fc['h'])
    layer.fc['z'] = np.zeros(layer.fs['z'])
    layer.fc['C'] = np.zeros(layer.fs['C'])

    layer.fc['A'] = np.zeros(layer.fs['A'])

    hp = np.zeros(layer.fs['ht'])
    C = np.zeros(layer.fs['Ct'])

    return X, hp, C
