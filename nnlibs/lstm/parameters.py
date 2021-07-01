#EpyNN/nnlibs/lstm/parameters.py
import nnlibs.commons.maths as cm

import numpy as np


def set_activation(layer):

    args = layer.activation
    layer.activate_input, layer.derivative_input = args[0], cm.get_derivative(args[0])
    layer.activate_forget, layer.derivative_forget = args[1], cm.get_derivative(args[1])
    layer.activate_memory, layer.derivative_memory = args[2], cm.get_derivative(args[2])
    layer.activate_candidate, layer.derivative_candidate = args[3], cm.get_derivative(args[3])
    layer.activate_output, layer.derivative_output = args[4], cm.get_derivative(args[4])

    return None


def init_shapes(layer,hidden_size,runData,output_size=None):
    ### Set layer dictionaries values
    ## Dimensions
    # Hidden size
    layer.d['h'] = hidden_size
    # Vocab size
    layer.d['v'] = runData.e['v']
    # z size
    layer.d['z'] = layer.d['h'] + layer.d['v']
    # Output size
    if output_size == None:
        output_size = layer.d['v']
    layer.d['o'] = output_size

    ## Forward pass shapes
    h1 =  ( layer.d['h'], 1 )
    hz = ( layer.d['h'], layer.d['z'] )
    oh = ( layer.d['o'], layer.d['h'] )
    o1 = ( layer.d['o'], 1 )

    # W, b - Forget gate
    layer.fs['Wf'],layer.fs['bf'] = hz, h1
    # W, b - Input gate
    layer.fs['Wi'],layer.fs['bi'] = hz, h1
    # W, b - Candidate gate
    layer.fs['Wg'],layer.fs['bg'] = hz, h1
    # W, b - Output gate
    layer.fs['Wo'],layer.fs['bo'] = hz, h1
    # W, b - Logit gate
    layer.fs['Wv'], layer.fs['bv'] = oh, o1

    return None


def init_params(layer):

    # Init W, b - Forget gate
    layer.p['Wf'] = np.random.randn(*layer.fs['Wf'])
    layer.p['bf'] = np.zeros(layer.fs['bf'])
    # Init W, b - Forget gate
    layer.p['Wi'] = np.random.randn(*layer.fs['Wi'])
    layer.p['bi'] = np.zeros(layer.fs['bi'])
    # Init W, b - Forget gate
    layer.p['Wg'] = np.random.randn(*layer.fs['Wg'])
    layer.p['bg'] = np.zeros(layer.fs['bg'])
    # Init W, b - Forget gate
    layer.p['Wv'] = np.random.randn(*layer.fs['Wv'])
    layer.p['bv'] = np.zeros(layer.fs['bv'])
    # Init W, b - Forget gate
    layer.p['Wo'] = np.random.randn(*layer.fs['Wo'])
    layer.p['bo'] = np.zeros(layer.fs['bo'])

    # Set init to False
    layer.init = False

    return None


def update_gradients(layer,t):
    # Number of sample
    m = layer.m = layer.fs['X'][-1]
    # Retrieve h (current t)
    h = layer.fc['h'][t]
    # Retrieve z (current t)
    z = layer.fc['z'][t]

    # Retrieve dv and update dWv and dbv
    dXt = layer.bc['dXt']
    layer.g['dWv'] += 1./ m * np.dot(dXt, h.T)
    layer.g['dbv'] += 1./ m * np.sum(dXt,axis=1,keepdims=True)

    # Retrieve do and update dWo and dbo
    do = layer.bc['do']
    layer.g['dWo'] += 1./ m * np.dot(do,z.T)
    layer.g['dbo'] += 1./ m * np.sum(do,axis=1,keepdims=True)
    # Retrieve dg and update dWg and dbg
    dg = layer.bc['dg']
    layer.g['dWg'] += 1./ m * np.dot(dg, z.T)
    layer.g['dbg'] += 1./ m * np.sum(dg,axis=1,keepdims=True)
    # Retrieve di and update dWi and dbi
    di = layer.bc['di']
    layer.g['dWi'] += 1./ m * np.dot(di, z.T)
    layer.g['dbi'] += 1./ m * np.sum(di,axis=1,keepdims=True)
    # Retrieve df and update dWf and dbf
    df = layer.bc['df']
    layer.g['dWf'] += 1./ m * np.dot(df, z.T)
    layer.g['dbf'] += 1./ m * np.sum(df,axis=1,keepdims=True)

    return None
