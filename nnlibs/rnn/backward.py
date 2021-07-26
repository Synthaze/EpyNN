# EpyNN/nnlibs/rnn/backward.py
# Related third party imports
import numpy as np


def rnn_backward(layer, dA):

    dX, dhn, dh = initialize_backward(layer, dA)

    # Loop through time steps
    for t in reversed(range(layer.d['t'])):

        if layer.binary == False:

            dXt = layer.bc['dXt'][t] = dX[t]

            dh = layer.bc['dh'] = np.dot(layer.p['W'].T, dXt) + dhn

        _dh = layer.bc['_dh'] = dh * layer.activate(layer.fc['h'][t], deriv=True)

        dhn = layer.bc['dhn'][t] = np.dot(layer.p['Vh'].T, _dh)

    return dA


def initialize_backward(layer, dA):

    dX = layer.bc['dX'] = dA

    layer.bc['dh'] = np.zeros(layer.fs['h'])
    layer.bc['dhn'] = np.zeros(layer.fs['h'])
    layer.bc['dXt'] = np.zeros_like(dX)

    dh = layer.bc['dh'] = np.dot(layer.p['W'].T, dX)

    dhn = np.zeros(layer.fs['ht'])

    return dX, dhn, dh
