# EpyNN/nnlibs/rnn/backward.py
# Related third party imports
import numpy as np


def initialize_backward(layer, dA):
    """.
    """
    #
    dX = layer.bc['dX'] = dA
    dh = layer.bc['dh'] = np.dot(layer.p['W'].T, dX)
    #
    layer.bc['dXs'] = np.zeros_like(dX)
    layer.bc['df'] = np.zeros(layer.fs['h'])
    layer.bc['dhn'] = np.zeros(layer.fs['h'])
    #
    dhn = np.zeros(layer.fs['hs'])

    return dX, dhn, dh


def rnn_backward(layer, dA):
    """.
    """
    # ()
    dX, dhn, dh = initialize_backward(layer, dA)

    # Step through reversed sequence
    for s in reversed(range(layer.d['s'])):

        if not layer.binary:
            # ()
            dXs = layer.bc['dXs'][s] = dX[s]
            # ()
            dh = layer.bc['dh'][s] = np.dot(layer.p['W'].T, dXs) + dhn

        # ()
        df = layer.bc['df'][s] = dh * layer.activate(layer.fc['h'][s], deriv=True)

        # ()
        dhn = layer.bc['dhn'][s] = np.dot(layer.p['Wh'].T, df)

    return dA    # To previous layer
