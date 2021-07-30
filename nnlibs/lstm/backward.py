# EpyNN/nnlibs/lstm/backward.py
# Related third party imports
import numpy as np


def initialize_backward(layer, dA):

    dX = layer.bc['dX'] = dA

    layer.bc['dh'] = np.zeros(layer.fs['h'])
    layer.bc['do'] = np.zeros_like(layer.bc['dh'])
    layer.bc['di'] = np.zeros_like(layer.bc['dh'])
    layer.bc['df'] = np.zeros_like(layer.bc['dh'])
    layer.bc['dg'] = np.zeros_like(layer.bc['dh'])
    layer.bc['dz'] = np.zeros_like(layer.bc['dh'])

    layer.bc['dhn'] = np.zeros(layer.fs['h'])
    layer.bc['dXt'] = np.zeros_like(dX)

    layer.bc['dC'] = np.zeros(layer.fs['C'])
    layer.bc['dCn'] = np.zeros(layer.fs['C'])
    layer.bc['dA'] = np.zeros(layer.bs['dA'])

    dhn = np.zeros_like(layer.bc['dh'][0])
    dCn = np.zeros_like(layer.bc['dC'][0])

    return dX, dhn, dCn


def lstm_backward(layer, dA):

    dX, dhn, dCn = initialize_backward(layer, dA)

    # Loop through steps
    for s in reversed(range(layer.d['h'])):

        dX = dX if layer.binary else layer.bc['dX'][s]

        dh = np.zeros_like(dhn) if layer.binary else dhn

        dh += np.dot(layer.p['W'].T, dX)

        do = dh * layer.activate_hidden(layer.fc['C'][s], deriv=True)
        do = layer.bc['do'][s] = do * layer.activate_output(layer.fc['o'][s], deriv=True)

        dC = layer.fc['o'][s] * dh * layer.activate_hidden(layer.activate_hidden(layer.fc['C'][s]), deriv=True)
        dC =  layer.bc['dC'][s] = dC + dCn

        dg = dC * layer.fc['i'][s]
        dg = layer.bc['dg'][s] = dg * layer.activate_candidate(layer.fc['g'][s], deriv=True)

        di = dC * layer.fc['g'][s]
        di = layer.bc['di'][s] = di * layer.activate_input(layer.fc['i'][s], deriv=True)

        df = dC * layer.fc['C'][s-1]
        df = layer.bc['df'][s] = df * layer.activate_forget(layer.fc['f'][s], deriv=True)

        dz = np.dot(layer.p['Wg'].T, dg)
        dz += np.dot(layer.p['Wo'].T, do)
        dz += np.dot(layer.p['Wi'].T, di)
        dz += np.dot(layer.p['Wf'].T, df)
        dz = layer.bc['dz'] = dz

        dhn = layer.bc['dhn'][s] = dz[:layer.d['h'], :]

        dCn = layer.bc['dCn'][s] = layer.fc['f'][s] * dC

        dA[s] = dz[:-layer.d['h'], :] * layer.fc['X'][s]

    return dA
