# EpyNN/nnlibs/rnn/forward.py
# Related third party imports
import numpy as np


def rnn_forward(layer, A):

    X, hp = initialize_forward(layer, A)

    # Loop through time steps
    for t in range(layer.d['t']):

        # Xt (X at current t) from X
        X = layer.fc['Xt'][t] = layer.fc['X'][:, t]

        # Calculate input gate
        h = np.dot(layer.p['Uh'], X)
        h += np.dot(layer.p['Vh'], hp) + layer.p['bh']
        h = hp = layer.fc['h'][t] = layer.activate_input(h)

        # Calculate output gate
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
    layer.fc['A'] = np.zeros(layer.fs['A'])

    hp = np.zeros(layer.fs['ht'])

    return X, hp
