# EpyNN/nnlibs/gru/forward.py
# Related third party imports
import numpy as np

def gru_forward(layer, A):

    X, hp = initialize_forward(layer, A)

    # Loop through time steps
    for t in range(layer.d['t']):

        X = layer.fc['Xt'][t] = layer.fc['X'][:, t]

        r = np.dot(layer.p['Wr'], X)
        r += np.dot(layer.p['Ur'], hp) + layer.p['br']
        r = layer.fc['r'][t] = layer.activate_reset(r)

        z = np.dot(layer.p['Wz'], X)
        z += np.dot(layer.p['Uz'], hp) + layer.p['bz']
        z = layer.fc['z'][t] = layer.activate_update(z)

        hh = np.dot(layer.p['Wh'], X)
        hh += np.dot(layer.p['Uh'], r * hp) + layer.p['bh']
        hh = layer.fc['hh'][t] = layer.activate_hidden(hh)

        h = hp = layer.fc['h'][t] = z*hp + (1-z)*hh

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
    layer.fc['hh'] = np.zeros_like(layer.fc['h'])
    layer.fc['z'] = np.zeros_like(layer.fc['h'])
    layer.fc['r'] = np.zeros_like(layer.fc['h'])
    layer.fc['A'] = np.zeros(layer.fs['A'])

    hp = np.zeros(layer.fs['ht'])

    return X, hp
