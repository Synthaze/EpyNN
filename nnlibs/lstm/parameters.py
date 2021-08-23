# EpyNN/nnlibs/lstm/parameters.py
# Related third party imports
import numpy as np


def lstm_compute_shapes(layer, A):
    """Compute forward shapes and dimensions for cells and layer.
    """
    X = A    # Input of current layer

    layer.fs['X'] = X.shape    # (m, s, v)

    layer.d['m'] = layer.fs['X'][0]    # Number of samples (m)
    layer.d['s'] = layer.fs['X'][1]    # Length of sequence (s)
    layer.d['v'] = layer.fs['X'][2]    # Vocabulary size (v)

    # Parameter Shapes
    vh = (layer.d['v'], layer.d['h'])
    hh = (layer.d['h'], layer.d['h'])
    h1 = (layer.d['h'],)

    # U applies to X - Gate/Activation-specific.
    layer.fs['Uf'] = layer.fs['Ui'] = layer.fs['Ug'] = layer.fs['Uo'] = vh
    # W applies to hp - Gate/Activation-specific.
    layer.fs['Wf'] = layer.fs['Wi'] = layer.fs['Wg'] = layer.fs['Wo'] = hh
    # b is added to the linear activation product - Gate/Activation-specific.
    layer.fs['bf'] = layer.fs['bi'] = layer.fs['bg'] = layer.fs['bo'] = h1

    # Shape of cache for hidden cell states (h) and memory cell states (C)
    msh = layer.fs['h'] = layer.fs['C'] = (layer.d['m'], layer.d['s'], layer.d['h'])


    return None


def lstm_initialize_parameters(layer):
    """Initialize parameters for layer.
    """
    # U, W, b - Forget gate
    layer.p['Uf'] = layer.initialization(layer.fs['Uf'], rng=layer.np_rng)
    layer.p['Wf'] = layer.initialization(layer.fs['Wf'], rng=layer.np_rng)
    layer.p['bf'] = np.zeros(layer.fs['bf'])

    # U, W, b - Input gate
    layer.p['Ui'] = layer.initialization(layer.fs['Ui'], rng=layer.np_rng)
    layer.p['Wi'] = layer.initialization(layer.fs['Wi'], rng=layer.np_rng)
    layer.p['bi'] = np.zeros(layer.fs['bi'])

    # U, W, b - Candidate activation
    layer.p['Ug'] = layer.initialization(layer.fs['Ug'], rng=layer.np_rng)
    layer.p['Wg'] = layer.initialization(layer.fs['Wg'], rng=layer.np_rng)
    layer.p['bg'] = np.zeros(layer.fs['bg'])

    # U, W, b - Output gate
    layer.p['Uo'] = layer.initialization(layer.fs['Uo'], rng=layer.np_rng)
    layer.p['Wo'] = layer.initialization(layer.fs['Wo'], rng=layer.np_rng)
    layer.p['bo'] = np.zeros(layer.fs['bo'])

    return None


def lstm_compute_gradients(layer):
    """Compute gradients with respect to weight and bias for cells and layer.
    """
    # Gradients initialization with respect to parameters
    for parameter in layer.p.keys():
        gradient = 'd' + parameter
        layer.g[gradient] = np.zeros_like(layer.p[parameter])

    # Iterate over reversed sequence steps
    for s in reversed(range(layer.d['s'])):

        # Retrieve from layer cache
        X = layer.fc['X'][:, s]
        hp = layer.fc['hp'][:, s]

        # (1) Gradients with respect to U, W, b
        do = layer.bc['do'][:, s]  # For output gate
        layer.g['dUo'] += np.dot(X.T, do)     # (1.1)
        layer.g['dWo'] += np.dot(hp.T, do)    # (1.2)
        layer.g['dbo'] += np.sum(do, axis=0)  # (1.3)

        # (2) Gradients with respect to U, W, b
        dg = layer.bc['dg'][:, s]  # For candidate activation
        layer.g['dUg'] += np.dot(X.T, dg)     # (2.1)
        layer.g['dWg'] += np.dot(hp.T, dg)    # (2.2)
        layer.g['dbg'] += np.sum(dg, axis=0)  # (2.3)

        # (3) Gradients with respect to U, W, b
        di = layer.bc['di'][:, s]  # For input gate
        layer.g['dUi'] += np.dot(X.T, di)     # (3.1)
        layer.g['dWi'] += np.dot(hp.T, di)    # (3.2)
        layer.g['dbi'] += np.sum(di, axis=0)  # (3.3)

        # (4) Gradients with respect to U, W, b
        df = layer.bc['df'][:, s]  # For forget gate
        layer.g['dUf'] += np.dot(X.T, df)     # (4.1)
        layer.g['dWf'] += np.dot(hp.T, df)    # (4.2)
        layer.g['dbf'] += np.sum(df, axis=0)  # (4.3)

    return None


def lstm_update_parameters(layer):
    """Update parameters for layer.
    """
    for gradient in layer.g.keys():
        parameter = gradient[1:]
        # Update is driven by learning rate and gradients
        layer.p[parameter] -= layer.lrate[layer.e] * layer.g[gradient]

    return None
