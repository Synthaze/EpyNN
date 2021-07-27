# EpyNN/nnlibs/gru/backward.py
# Related third party imports
import numpy as np


def gru_backward(layer, dA):

    dX, dhn, dXt, dh, dhh = initialize_backward(layer, dA)

    # Loop through time steps
    for t in reversed(range(layer.d['t'])):

        if layer.binary == False:

            dXt = layer.bc['dXt'][t] = dX[t]

            dh = layer.bc['dh'][t] = np.dot(layer.p['W'].T, dXt) + dhn

            dhh = dh * (1-layer.fc['z'][t])
            dhh = layer.bc['dhh'][t] = dhh * layer.activate_hidden(layer.fc['hh'][t], deriv=True)

        dr = np.dot(layer.p['Uh'].T, dhh)
        dr = dr * layer.fc['h'][t-1]
        dr = layer.bc['dr'][t] = dr * layer.activate_reset(layer.fc['r'][t], deriv=True)

        dz = np.multiply(dh, layer.fc['h'][t-1] - layer.fc['hh'][t])
        dz = layer.bc['dz'][t] = dz * layer.activate_update(layer.fc['z'][t], deriv=True)

        dhfhh = np.dot(layer.p['Uh'].T, dhh)
        dhfhh = layer.bc['dhfhh'][t] = dhfhh * layer.fc['r'][t]
        dhfr = layer.bc['dhfr'][t] = np.dot(layer.p['Ur'].T, dr)
        dhfzi = layer.bc['dhfzi'][t] = np.dot(layer.p['Uz'].T, dz)
        dhfz = layer.bc['dhfz'][t] = dh * layer.fc['z'][t]
        dhn = layer.bc['dhn'][t] = dhfhh + dhfr + dhfzi + dhfz

    return dA


def initialize_backward(layer, dA):

    dX = layer.bc['dX'] = dA

    dXt = layer.bc['dXt'] = layer.bc['dX']

    layer.bc['dh'] = np.zeros(layer.fs['h'])
    layer.bc['dhh'] = np.zeros(layer.fs['h'])
    layer.bc['dr'] = np.zeros_like(layer.bc['dh'])
    layer.bc['dz'] = np.zeros_like(layer.bc['dh'])

    layer.bc['dhfhh'] = np.zeros_like(layer.bc['dh'])
    layer.bc['dhfr'] = np.zeros_like(layer.bc['dh'])
    layer.bc['dhfzi'] = np.zeros_like(layer.bc['dh'])
    layer.bc['dhfz'] = np.zeros_like(layer.bc['dh'])
    layer.bc['dhn'] = np.zeros(layer.fs['h'])

    dh = layer.bc['dh'] = np.dot(layer.p['W'].T, dXt)

    dhh = dh * (1-layer.fc['z'][-1])
    dhh = dhh * layer.activate_input(layer.fc['hh'][-1], deriv=True)

    dhn = np.zeros(layer.fs['ht'])

    return dX, dhn, dXt, dh, dhh
