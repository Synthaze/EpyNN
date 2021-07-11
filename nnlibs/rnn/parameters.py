#EpyNN/nnlibs/rnn/parameters.py
import nnlibs.commons.maths as cm

import numpy as np


def set_activation(layer):

    args = layer.activation
    layer.activate_input, layer.derivative_input = args[0], cm.get_derivative(args[0])
    layer.activate_output, layer.derivative_output = args[1], cm.get_derivative(args[1])

    return None


def init_shapes(layer):
    ### Set layer dictionaries values
    ## Dimensions
    # Hidden size
    layer.d['h'] = layer.hidden_size
    # Vocab size
    layer.d['v'] = layer.vocab_size = layer.d['v']
    # Output size
    if layer.binary == False:
        output_size = layer.d['v']
    else:
        output_size = 2

    layer.d['o'] = layer.output_size = output_size

    ## Forward pass shapes
    hv = ( layer.d['h'], layer.d['v'] )
    hh = ( layer.d['h'], layer.d['h'] )
    h1 = ( layer.d['h'], 1 )
    oh = ( layer.d['o'], layer.d['h'] )
    o1 = ( layer.d['o'], 1 )

    # W, U, b - _ gate
    layer.fs['U'], layer.fs['V'], layer.fs['bh'] = hv, hh, h1
    # W, U, b - _ gate
    layer.fs['W'], layer.fs['bo'] = oh, o1

    return None


def init_forward(layer,A):

    # Cache X (current) from A (prev)
    X = layer.fc['X'] = A
    layer.fs['X'] = X.shape

    # Init cache shapes
    layer.ts = layer.fs['X'][1]

    for c in ['h','A']:
        layer.fc[c] = [0] * layer.ts

    # Init h
    layer.fs['h'] = ( layer.d['h'],layer.fs['X'][-1] )
    # Init h
    hp = np.zeros(layer.fs['h'])

    return X, hp


def end_forward(layer):

    layer.fc = { k:np.array(v) for k,v in layer.fc.items() }

    A = layer.fc['A']

    if layer.binary == True:
        A = layer.fc['A'] = A[-1]

    return A


def init_backward(layer,dA):

    # Cache X (current) from A (prev)
    dX = layer.bc['dX'] = dA
    # Init dh_next (dhn)
    dhn = layer.bc['dhn'] = np.zeros_like(layer.fc['h'][0])

    # Cache dXt (dX at current t) from dX
    dXt = layer.bc['dXt'] = layer.bc['dX']
    # Cache dh (current) from dXt (prev), dhn, z (current) and h (current)
    dh = layer.bc['dh'] = np.dot(layer.p['W'].T, dXt)

    return dX, dhn, dXt, dh


def init_params(layer):

    # W, U, b - _ gate
    layer.p['U'] = layer.initialization(layer.fs['U'])
    layer.p['V'] = layer.initialization(layer.fs['V'])
    layer.p['bh'] = np.zeros(layer.fs['bh'])

    # W, U, b - _ gate
    layer.p['W'] = layer.initialization(layer.fs['W'])
    layer.p['bo'] = np.zeros(layer.fs['bo'])

    # Set init to False
    layer.init = False

    return None


def update_gradients(layer,t):
    # Number of sample
    m = layer.m = layer.fs['X'][-1]
    # Retrieve h (current t)
    h = layer.fc['h'][t]
    hp = layer.fc['h'][t-1]
    # Retrieve Xt
    Xt = layer.fc['X'][:,t]

    # _
    dXt = layer.bc['dXt']
    layer.g['dW'] += 1./ m * np.dot(dXt,h.T)
    layer.g['dbo'] += 1./ m * np.sum(dXt,axis=1,keepdims=True)

    # _
    df = layer.bc['df']
    layer.g['dU'] += 1./ m * np.dot(df, Xt.T)
    layer.g['dV'] += 1./ m * np.dot(df, hp.T)
    layer.g['dbh'] += 1./ m * np.sum(df,axis=1,keepdims=True)


    return None
