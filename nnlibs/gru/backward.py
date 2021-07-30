# EpyNN/nnlibs/gru/backward.py
# Related third party imports
import numpy as np


def initialize_backward(layer, dA):

    dX = layer.bc['dX'] = dA

    layer.bc['dh'] = np.zeros(layer.fs['h'])
    layer.bc['dhh'] = np.zeros(layer.fs['h'])
    layer.bc['dr'] = np.zeros_like(layer.bc['dh'])
    layer.bc['dz'] = np.zeros_like(layer.bc['dh'])

    layer.bc['dhfhh'] = np.zeros_like(layer.bc['dh'])
    layer.bc['dhfr'] = np.zeros_like(layer.bc['dh'])
    layer.bc['dhfzi'] = np.zeros_like(layer.bc['dh'])
    layer.bc['dhfz'] = np.zeros_like(layer.bc['dh'])
    layer.bc['dhn'] = np.zeros(layer.fs['h'])

    dhn = np.zeros_like(layer.bc['dh'][0])

    return dX, dhn


def gru_backward(layer, dA):

    dX, dhn = initialize_backward(layer, dA)

    # Loop through steps
    for s in reversed(range(layer.d['h'])):

        dX = dX if layer.binary else layer.bc['dX'][s]

        dh = np.zeros_like(dhn) if layer.binary else dhn

        dh += np.dot(layer.p['W'].T, dX)

        if layer.binary == False:

            dhh = dh * (1-layer.fc['z'][s])
            dhh = layer.bc['dhh'][s] = dhh * layer.activate_hidden(layer.fc['hh'][s], deriv=True)

        dr = np.dot(layer.p['Uh'].T, dhh)
        dr = dr * layer.fc['h'][s - 1]
        dr = layer.bc['dr'][s] = dr * layer.activate_reset(layer.fc['r'][s], deriv=True)

        dz = np.multiply(dh, layer.fc['h'][s - 1] - layer.fc['hh'][s])
        dz = layer.bc['dz'][s] = dz * layer.activate_update(layer.fc['z'][s], deriv=True)

        dhfhh = np.dot(layer.p['Uh'].T, dhh)
        dhfhh = layer.bc['dhfhh'][s] = dhfhh * layer.fc['r'][s]
        dhfr = layer.bc['dhfr'][s] = np.dot(layer.p['Ur'].T, dr)
        dhfzi = layer.bc['dhfzi'][s] = np.dot(layer.p['Uz'].T, dz)
        dhfz = layer.bc['dhfz'][s] = dh * layer.fc['z'][s]
        dhn = layer.bc['dhn'][s] = dhfhh + dhfr + dhfzi + dhfz

        dA[s] = layer.fc['X'][s] * dX

    return dA
