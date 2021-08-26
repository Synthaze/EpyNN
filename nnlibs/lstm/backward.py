# EpyNN/nnlibs/lstm/backward.py
# Related third party imports
import numpy as np


def initialize_backward(layer, dX):
    """Backward cache initialization.

    :param layer: An instance of LSTM layer.
    :type layer: :class:`nnlibs.lstm.models.LSTM`

    :param dX: Output of backward propagation from next layer.
    :type dX: :class:`numpy.ndarray`

    :return: Input of backward propagation for current layer.
    :rtype: :class:`numpy.ndarray`

    :return: Next cell state initialized with zeros.
    :rtype: :class:`numpy.ndarray`

    :return: Next memory state initialized with zeros.
    :rtype: :class:`numpy.ndarray`
    """
    if layer.sequences:
        dA = dX                         # Full length sequence
    elif not layer.sequences:
        dA = np.zeros(layer.fs['h'])    # Empty full length sequence
        dA[:, -1] = dX                  # Assign to last index

    cache_keys = ['dh', 'do', 'dg', 'di', 'df', 'dz', 'dC', 'dCn', 'dhn']
    layer.bc.update({k: np.zeros(layer.fs['h']) for k in cache_keys})

    layer.bc['dA'] = dA
    layer.bc['dX'] = np.zeros(layer.fs['X'])    # To previous layer

    dhn = layer.bc['dhn'][:, 0]                 # To previous cell
    dCn = layer.bc['dCn'][:, 0]                 # To previous cell

    return dA, dhn, dCn


def lstm_backward(layer, dX):
    """Backward propagate error gradients through LSTM cells to previous layer.
    """
    # (1) Initialize cache, hidden and memory cell state gradients
    dA, dhn, dCn = initialize_backward(layer, dX)

    # Reverse iteration over sequence steps
    for s in reversed(range(layer.d['s'])):

        # (2s) Slice sequence (m, s, u) with respect to step
        dA = layer.bc['dA'][:, s]

        # (3s) Gradient of the loss with respect to hidden cell state
        dh = dA      # Grad. from next layer (dA)
        dh += dhn    # Grad. from next cell (dhn)

        # (4s) Gradient of the loss w.r.t output gate
        do = dh * layer.activate(layer.fc['C'][:, s])
        do = layer.bc['do'][:, s] = do * layer.activate_output(layer.fc['o'][:, s], linear=False, deriv=True)

        # (5s) Gradient of the loss w.r.t memory cell state
        dC = dh * layer.fc['o'][:, s] * layer.activate(layer.fc['C'][:, s], deriv=True)
        dC =  layer.bc['dC'][:, s] = dC + dCn

        # (6.1s) Gradient of the loss w.r.t candidate
        dg = dC * layer.fc['i'][:, s]
        dg = layer.bc['dg'][:, s] = dg * layer.activate_candidate(layer.fc['g'][:, s], linear=False, deriv=True)

        # (6.2s) Gradient of the loss w.r.t input gate
        di = dC * layer.fc['g'][:, s]
        di = layer.bc['di'][:, s] = di * layer.activate_input(layer.fc['i'][:, s], linear=False, deriv=True)

        # (7s) Gradient of the loss w.r.t forget gate
        df = dC * layer.fc['Cp'][:, s]
        df = layer.bc['df'][:, s] = df * layer.activate_forget(layer.fc['f'][:, s], linear=False, deriv=True)

        # (8s) Gradient of the loss w.r.t previous hidden state
        dhn = np.dot(do, layer.p['Wo'].T)
        dhn += np.dot(dg, layer.p['Wg'].T)
        dhn += np.dot(di, layer.p['Wi'].T)
        dhn += np.dot(df, layer.p['Wf'].T)

        dhn = layer.bc['dhn'][:, s] = dhn[:, :layer.d['u']]       # To previous cell

        # (9s) Gradient of the loss w.r.t previous memory state
        dCn = layer.bc['dCn'][:, s] = layer.fc['f'][:, s] * dC    # To previous cell

        # (10s) Gradient of the loss w.r.t X
        dX = np.dot(dg, layer.p['Ug'].T)
        dX += np.dot(do, layer.p['Uo'].T)
        dX += np.dot(di, layer.p['Ui'].T)
        dX += np.dot(df, layer.p['Uf'].T)
        layer.bc['dX'][:, s] = dX

    dX = layer.bc['dX']

    return dX    # To previous layer
