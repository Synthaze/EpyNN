#EpyNN/nnlibs/gru/backward.py
import nnlibs.meta.parameters as mp

import nnlibs.gru.parameters as gp

import numpy as np


def gru_backward(layer,dA):

    # Cache X (current) from A (prev)
    dX = layer.bc['dX'] = dA
    # Init dh_next (dhn)
    dhn = layer.bc['dh_n'] = np.zeros_like(layer.fc['h'][0])

    # Loop through time steps
    for t in reversed(range(layer.fs['X'][1])):

        # Cache dXt (dX at current t) from dX
        dXt = layer.bc['dXt'] = dX[t]
        # Cache dh (current) from dXt (prev), dhn, z (current) and h (current)
        dh = np.dot( layer.p['Wy'].T, dXt ) + dhn
        dh = np.multiply( dh, (1-layer.fc['z'][t]) )
        dh = layer.bc['dh'] = layer.derivative_input(dh,layer.fc['h'][t])
        # Cache dh (current) from dXt (prev), dhn, z (current) and h (current)
        dr = np.dot(layer.p['Uh'].T, dh)
        dr = np.multiply(dr, layer.fc['h'][t-1])
        dr = layer.bc['dr'] = layer.derivative_reset(dr,layer.fc['r'][t])
        # Cache dh (current) from dXt (prev), dhn, z (current) and h (current)
        dz = np.multiply( dh, layer.fc['h'][t-1] - layer.fc['h'][t] )
        dz = layer.bc['dz'] = layer.derivative_update( dz, layer.fc['z'][t] )

        # Update gradients
        gp.update_gradients(layer,t)

        # All influences of previous layer to loss
        # Cache dhfhh
        dhfhh = np.dot( layer.p['Uh'].T, layer.bc['dh'] )
        dhfhh = layer.bc['dhfhh'] = np.multiply(dhfhh, layer.fc['r'][t])
        # Cache dhfr
        dhfr = layer.bc['dhfr'] = np.dot( layer.p['Ur'].T, layer.bc['dr'] )
        # Cache dhfzi
        dhfzi = layer.bc['dhfzi'] = np.dot( layer.p['Uz'].T, layer.bc['dz'] )
        # Cache dhfz
        dhfz = layer.bc['dhfz'] = np.multiply(layer.bc['dh'], layer.fc['z'][t])
        # Cache dhn
        layer.bc['dhn'] = dhn = dhfhh + dhfr + dhfzi + dhfz

    # Clip gradients
    mp.clip_gradient(layer)

    return None
