# EpyNN/nnlibs/gru/forward.py
# Related third party imports
import numpy as np


def initialize_forward(layer, A):

    X = layer.fc['X'] = np.zeros(layer.fs['X'])

    X[:layer.d['s']] = A

    layer.fc['h'] = np.zeros(layer.fs['h'])
    layer.fc['hh'] = np.zeros_like(layer.fc['h'])
    layer.fc['z'] = np.zeros_like(layer.fc['h'])
    layer.fc['r'] = np.zeros_like(layer.fc['h'])
    layer.fc['A'] = np.zeros(layer.fs['A'])

    hp = np.zeros_like(layer.fc['h'][0])

    return X, hp


def gru_forward(layer, A):
    """Forward propagate signal to next layer.
    """
    X, hp = initialize_forward(layer, A)

    # Loop through steps
    for s in range(layer.d['h']):

        X = layer.fc['X'][s]

        r = np.dot(layer.p['Wr'], X)
        r += np.dot(layer.p['Ur'], hp) + layer.p['br']
        r = layer.fc['r'][s] = layer.activate_reset(r)

        z = np.dot(layer.p['Wz'], X)
        z += np.dot(layer.p['Uz'], hp) + layer.p['bz']
        z = layer.fc['z'][s] = layer.activate_update(z)

        hh = np.dot(layer.p['Wh'], X)
        hh += np.dot(layer.p['Uh'], r * hp) + layer.p['bh']
        hh = layer.fc['hh'][s] = layer.activate_hidden(hh)

        h = hp = layer.fc['h'][s] = z*hp + (1-z)*hh

        A = np.dot(layer.p['W'], h) + layer.p['b']
        A = layer.fc['A'][s] = layer.activate(A)

    # Return layer.fc['A'] if layer.binary else A
    A = terminate_forward(layer)

    return A   # To next layer


def terminate_forward(layer):
    """Check if many-to-many or many-to-one (binary) mode.

    :param layer: GRU layer object
    :type layer: class:`nnlibs.gru.models.GRU`

    :return: Output of forward propagation for current layer
    :rtype: class:`numpy.ndarray`
    """
    layer.fc['A'] = layer.fc['A'][-1] if layer.binary else layer.fc['A']

    A = layer.fc['A']

    return A
