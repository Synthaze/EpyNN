# EpyNN/nnlibs/lstm/forward.py
# Related third party imports
import numpy as np


def initialize_forward(layer, A):

    X = layer.fc['X'] = np.zeros(layer.fs['X'])

    X[:layer.d['s']] = A

    layer.fc['h'] = np.zeros(layer.fs['h'])
    layer.fc['o'] = np.zeros_like(layer.fc['h'])
    layer.fc['i'] = np.zeros_like(layer.fc['h'])
    layer.fc['f'] = np.zeros_like(layer.fc['h'])
    layer.fc['g'] = np.zeros_like(layer.fc['h'])
    layer.fc['z'] = np.zeros(layer.fs['z'])
    layer.fc['C'] = np.zeros(layer.fs['C'])

    layer.fc['A'] = np.zeros(layer.fs['A'])

    hp = np.zeros_like(layer.fc['h'][0])
    C = np.zeros_like(layer.fc['C'][0])

    return X, hp, C


def lstm_forward(layer, A):

    X, hp, C = initialize_forward(layer, A)

    # Loop through time steps
    for s in range(layer.d['h']):

        # Xt (X at current t) from X
        X = layer.fc['X'][s]

        # Concatenate input and hidden state
        z = layer.fc['z'][s] = np.row_stack((hp, X))

        # Calculate forget gate
        f = np.dot(layer.p['Wf'], z) + layer.p['bf']
        f = layer.fc['f'][s] = layer.activate_forget(f)

        # Calculate input gate
        i = np.dot(layer.p['Wi'], z) + layer.p['bi']
        i = layer.fc['i'][s] = layer.activate_input(i)

        # Calculate candidate
        g = np.dot(layer.p['Wg'], z) + layer.p['bg']
        g = layer.fc['g'][s] = layer.activate_candidate(g)

        # Calculate output gate
        o = np.dot(layer.p['Wo'], z) + layer.p['bo']
        o = layer.fc['o'][s] = layer.activate_output(o)

        # Calculate memory state (prev)
        C = layer.fc['C'][s] = f*C + i*g

        # Calculate hidden state
        h = layer.fc['h'][s] = o * layer.activate_hidden(C)

        # Calculate logits
        A = np.dot(layer.p['W'], h) + layer.p['b']
        A = layer.fc['A'][s] = layer.activate(A)

    # Return layer.fc['A'] if layer.binary else A
    A = terminate_forward(layer)

    return A


def terminate_forward(layer):
    """.
    """
    if layer.binary:
        A = layer.fc['A'] = layer.fc['A'][-1]
    else:
        A = layer.fc['A']

    return A
