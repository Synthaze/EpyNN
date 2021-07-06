#EpyNN/nnlibs/lstm/backward.py
import nnlibs.meta.parameters as mp

import nnlibs.lstm.parameters as lp

import numpy as np


def lstm_backward(layer,dA):

    dX, dhn, dCn, dXt, dh = lp.init_backward(layer,dA)

    # Loop through time steps
    for t in reversed(range(layer.ts)):

        if layer.binary == False:
            # Cache dXt (dX at current t) from dX
            dXt = layer.bc['dXt'] = dX[:,t]
            # Cache dh (current) from dXt (prev) and dhn
            dh = layer.bc['dh'] = np.dot(layer.p['Wv'].T, dXt) + dhn

        # Cache do (current) from dh (current) and c (current)
        do = dh * layer.activate_hidden(layer.fc['C'][t])
        do = layer.bc['do'] = layer.derivative_output(do,layer.fc['o'][t])
        # Cache dC (current) from o (current) and dh (current) and c (current)
        dC = layer.fc['o'][t] * layer.derivative_hidden(dh,layer.activate_hidden(layer.fc['C'][t]))
        dC =  layer.bc['dC'] = dC + dCn
        # Cache dg (current) from dC (current) and i (current) and g (current)
        dg = dC * layer.fc['i'][t]
        dg = layer.bc['dg'] = layer.derivative_candidate(dg,layer.fc['g'][t])
        # Cache dg (current) from dC (current) and i (current) and g (current)
        di = dC * layer.fc['g'][t]
        di = layer.bc['di'] = layer.derivative_input(di,layer.fc['i'][t])
        # Cache df (current) from dC (current) and Cp (current) and f (current)
        df = dC * layer.fc['C'][t-1]
        df = layer.bc['df'] = layer.derivative_forget(df,layer.fc['f'][t])

        # Update gradients
        lp.update_gradients(layer,t)

        # Cache dz (current) from do, dg, di, df
        dz = np.dot(layer.p['Wg'].T, dg)
        dz += np.dot(layer.p['Wo'].T, do)
        dz += np.dot(layer.p['Wi'].T, di)
        dz += np.dot(layer.p['Wf'].T, df)
        dz = layer.bc['dz'] = dz
        # Cache dhp from dz
        dhn = layer.bc['dhn'] = dz[:layer.d['h'], :]
        # Cache dCp from f (current) and  dC (current)
        dCn = layer.bc['dCn'] = layer.fc['f'][t] * dC

    # Clip gradients
    mp.clip_gradient(layer)

    return None
