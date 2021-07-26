# EpyNN/nnlibs/lstm/backward.py
# Related third party imports
import numpy as np


def lstm_backward(layer, dA):

    dX, dhn, dCn, dXt, dh = initialize_backward(layer, dA)

    # Loop through time steps
    for t in reversed(range(layer.d['t'])):

        if layer.binary == False:

            dXt = layer.bc['dXt'] = dX[t]

            dh = layer.bc['dh'][t] = np.dot(layer.p['W'].T, dXt) + dhn

        do = dh * layer.activate_hidden(layer.fc['C'][t], deriv=True)
        do = layer.bc['do'][t] = do * layer.activate_output(layer.fc['o'][t], deriv=True)

        dC = layer.fc['o'][t] * dh * layer.activate_hidden(layer.activate_hidden(layer.fc['C'][t]), deriv=True)
        dC =  layer.bc['dC'][t] = dC + dCn

        dg = dC * layer.fc['i'][t]
        dg = layer.bc['dg'][t] = dg * layer.activate_candidate(layer.fc['g'][t], deriv=True)

        di = dC * layer.fc['g'][t]
        di = layer.bc['di'][t] = di * layer.activate_input(layer.fc['i'][t], deriv=True)

        df = dC * layer.fc['C'][t-1]
        df = layer.bc['df'][t] = df * layer.activate_forget(layer.fc['f'][t], deriv=True)

        dz = np.dot(layer.p['Wg'].T, dg)
        dz += np.dot(layer.p['Wo'].T, do)
        dz += np.dot(layer.p['Wi'].T, di)
        dz += np.dot(layer.p['Wf'].T, df)
        dz = layer.bc['dz'] = dz

        dhn = layer.bc['dhn'][t] = dz[:layer.d['h'], :]

        dCn = layer.bc['dCn'][t] = layer.fc['f'][t] * dC

    return dA


def initialize_backward(layer, dA):

    dX = layer.bc['dX'] = dA

    layer.bc['dh'] = np.zeros(layer.fs['h'])
    layer.bc['do'] = np.zeros_like(layer.bc['dh'])
    layer.bc['di'] = np.zeros_like(layer.bc['dh'])
    layer.bc['df'] = np.zeros_like(layer.bc['dh'])
    layer.bc['dg'] = np.zeros_like(layer.bc['dh'])

    layer.bc['dhn'] = np.zeros(layer.fs['h'])
    layer.bc['dXt'] = np.zeros_like(dX)

    layer.bc['dC'] = np.zeros(layer.fs['C'])
    layer.bc['dCn'] = np.zeros(layer.fs['C'])

    dXt = layer.bc['dXt'] = layer.bc['dX']
    dh = layer.bc['dh'] = np.dot(layer.p['W'].T, dXt)

    dhn = np.zeros(layer.fs['ht'])
    dCn = np.zeros(layer.fs['Ct'])

    return dX, dhn, dCn, dXt, dh
