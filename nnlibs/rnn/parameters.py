#EpyNN/nnlibs/rnn/parameters.py
import nnlibs.commons.maths as cm

import numpy as np


def set_activation(layer):

    args = layer.activation
    layer.activate_input, layer.derivative_input = args[0], cm.get_derivative(args[0])
    layer.activate_output, layer.derivative_output = args[1], cm.get_derivative(args[1])

    return None



def init_shapes(layer,hidden_size,runData,output_size=None):
    ### Set layer dictionaries values
    ## Dimensions
    # Hidden size
    layer.d['h'] = hidden_size
    # Vocab size
    layer.d['v'] = runData.e['v']
    # Output size
    if output_size == None:
        output_size = layer.d['v']
    layer.d['o'] = output_size

    ## Forward pass shapes
    # W, U, b - _ gate
    layer.fs['U'] = hv = ( layer.d['h'], layer.d['v'] )
    layer.fs['V'] = hh = ( layer.d['h'], layer.d['h'] )
    layer.fs['bh'] = h1 = ( layer.d['h'], 1 )
    # W, U, b - _ gate
    layer.fs['W'] = oh = ( layer.d['o'], layer.d['h'] )
    layer.fs['bo'] = o1 = ( layer.d['o'], 1 )

    return None


def init_params(layer):

    # W, U, b - _ gate
    layer.p['U'] = np.random.randn(*layer.fs['U']) * 0.01
    layer.p['V'] = np.random.randn(*layer.fs['V']) * 0.01
    layer.p['bh'] = np.zeros(layer.fs['bh'])
    # W, U, b - _ gate
    layer.p['W'] = np.random.randn(*layer.fs['W']) * 0.01
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
    do = layer.bc['dXt']
    layer.g['dW'] += 1./ m * np.dot(do,h.T)
    layer.g['dbo'] += 1./ m * np.sum(do,axis=1,keepdims=True)

    # _
    df = layer.bc['df']
    layer.g['dU'] += 1./ m * np.dot(df, Xt.T)
    layer.g['dV'] += 1./ m * np.dot(df, hp.T)
    layer.g['dbh'] += 1./ m * np.sum(df,axis=1,keepdims=True)


    return None
