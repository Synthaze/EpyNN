# EpyNN/nnlibs/rnn/forward.py
# Related third party imports
import numpy as np


def initialize_forward(layer, A):
    """.
    """
    X = layer.fc['X'] = A

    layer.fc['h'] = np.zeros(layer.fs['h'])
    layer.fc['A'] = np.zeros(layer.fs['A'])

    hp = np.zeros_like(layer.fc['h'][0])

    return X, hp


def rnn_forward(layer, A):
    """.
    """
    # (1) Initialize cache and hidden cell state
    X, hp = initialize_forward(layer, A)

    # Step through sequence
    for s in range(layer.d['s']):

        # (2s) Slice sequence (v, s, m) with respect to step
        X = layer.fc['X'][:, s]    # (v, m)

        # (3s) Compute hidden cell state
        h = np.dot(layer.p['Wx'], X)
        h += np.dot(layer.p['Wh'], hp) + layer.p['bh']
        h = hp = layer.fc['h'][s] = layer.activate_hidden(h)

        # (4s) Compute cell output to next layer
        A = np.dot(layer.p['W'], h) + layer.p['b']
        A = layer.fc['A'][s] = layer.activate(A)

    # Return layer.fc['A'] if layer.binary else A
    A = terminate_forward(layer)

    return A    # To next layer


def terminate_forward(layer):
    """.
    """
    if layer.binary:
        A = layer.fc['A'] = layer.fc['A'][-1]
    else:
        A = layer.fc['A']

    return A
