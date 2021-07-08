#EpyNN/nnlibs/gru/backward.py
import nnlibs.meta.parameters as mp

import nnlibs.gru.parameters as gp

import numpy as np


def gru_backward(layer,dA):

    dX, dhn, dXt, dh, dhh = gp.init_backward(layer,dA)

    # Loop through time steps
    for t in reversed(range(layer.ts)):

        if layer.binary == False:
            # Cache dXt (dX at current t) from dX
            dXt = layer.bc['dXt'] = dX[t]
            # Cache dh (current) from dXt (prev), dhn, z (current) and h (current)
            dh = layer.bc['dh'] = np.dot( layer.p['Wy'].T, dXt ) + dhn
            dhh = np.multiply( dh, (1-layer.fc['z'][t]) )
            dhh = layer.bc['dhh'] = layer.derivative_input(dhh,layer.fc['hh'][t])

        # Cache dh (current) from dXt (prev), dhn, z (current) and h (current)
        dr = np.dot(layer.p['Uh'].T, dhh)
        dr = np.multiply(dr, layer.fc['h'][t-1])
        dr = layer.bc['dr'] = layer.derivative_reset(dr,layer.fc['r'][t])
        # Cache dh (current) from dXt (prev), dhn, z (current) and h (current)
        dz = np.multiply( dh, layer.fc['h'][t-1] - layer.fc['hh'][t] )
        dz = layer.bc['dz'] = layer.derivative_update( dz, layer.fc['z'][t] )

        # Update gradients
        gp.update_gradients(layer,t)

        # All influences of previous layer to loss
        # Cache dhfhh
        dhfhh = np.dot( layer.p['Uh'].T, dhh )
        dhfhh = layer.bc['dhfhh'] = np.multiply(dhfhh, layer.fc['r'][t])
        # Cache dhfr
        dhfr = layer.bc['dhfr'] = np.dot( layer.p['Ur'].T, dr )
        # Cache dhfzi
        dhfzi = layer.bc['dhfzi'] = np.dot( layer.p['Uz'].T, dz )
        # Cache dhfz
        dhfz = layer.bc['dhfz'] = np.multiply(dh, layer.fc['z'][t])
        # Cache dhn
        dhn = layer.bc['dhn'] = dhfhh + dhfr + dhfzi + dhfz

    # Clip gradients
    mp.clip_gradient(layer)

    return None
