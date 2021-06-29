#EpyNN/nnlibs/gru/parameters.py
import numpy as np


def init_params(layer):

    layer.p['Wz'] = np.random.randn( layer.s['Wz'][0],layer.s['Wz'][1] )
    layer.p['Uz'] = np.random.randn( layer.s['Uz'][0],layer.s['Uz'][1] )
    layer.p['Wr'] = np.random.randn( layer.s['Wr'][0],layer.s['Wr'][1] )
    layer.p['Ur'] = np.random.randn( layer.s['Ur'][0],layer.s['Ur'][1] )
    layer.p['Wh'] = np.random.randn( layer.s['Wh'][0],layer.s['Wh'][1] )
    layer.p['Uh'] = np.random.randn( layer.s['Uh'][0],layer.s['Uh'][1] )
    layer.p['Wy'] = np.random.randn( layer.s['Wy'][0],layer.s['Wy'][1] )

    layer.p['bz'] = np.zeros( layer.s['bz'] )
    layer.p['br'] = np.zeros( layer.s['br'] )
    layer.p['bh'] = np.zeros( layer.s['bh'] )
    layer.p['by'] = np.zeros( layer.s['by'] )

    layer.init = False

    return None
