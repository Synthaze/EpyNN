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
    """Backward propagate error gradients to previous layer.
    """
    # (1) Initialize cache, hidden and memory cell state gradients
    dA, dhn, dCn = initialize_backward(layer, dX)

    # Reverse iteration over sequence steps
    for s in reversed(range(layer.d['s'])):

        # (2s) Slice sequence (m, s, u) with respect to step
        dA = layer.bc['dA'][:, s]

        # (3s) Gradient of the loss with respect to hidden cell state
        dh = layer.bc['dh'][:, s] = (
            (dA + dhn)
        )

        # (4s) Gradient of the loss w.r.t output gate
        do = layer.bc['do'][:, s] = (
            dh
            * layer.activate(layer.fc['C'][:, s])
            * layer.activate_output(layer.fc['o_'][:, s], deriv=True)
        )

        # (5s) Gradient of the loss w.r.t memory cell state
        dC = layer.bc['dC'][:, s] = (
            dh
            * layer.fc['o'][:, s]
            * layer.activate(layer.fc['C'][:, s], deriv=True)
            + dCn
        )

        # (6.1s) Gradient of the loss w.r.t candidate
        dg = layer.bc['dg'][:, s] = (
            dC
            * layer.fc['i'][:, s]
            * layer.activate_candidate(layer.fc['g_'][:, s], deriv=True)
        )

        # (6.2s) Gradient of the loss w.r.t input gate
        di = layer.bc['di'][:, s] = (
            dC
            * layer.fc['g'][:, s]
            * layer.activate_input(layer.fc['i_'][:, s], deriv=True)
        )

        # (7s) Gradient of the loss w.r.t forget gate
        df = layer.bc['df'][:, s] = (
            dC
            * layer.fc['Cp'][:, s]
            * layer.activate_forget(layer.fc['f_'][:, s], deriv=True)
        )

        # (8s) Gradient of the loss w.r.t previous memory state
        dCn = layer.bc['dCn'][:, s] = (
            dC
            * layer.fc['f'][:, s]
        )

        # (9s) Gradient of the loss w.r.t previous hidden state
        dhn = layer.bc['dhn'][:, s] = (
            np.dot(do, layer.p['Wo'].T)
            + np.dot(dg, layer.p['Wg'].T)
            + np.dot(di, layer.p['Wi'].T)
            + np.dot(df, layer.p['Wf'].T)
        )

        # (10s) Gradient of the loss w.r.t X
        dX = layer.bc['dX'][:, s] = (
            np.dot(dg, layer.p['Ug'].T)
            + np.dot(do, layer.p['Uo'].T)
            + np.dot(di, layer.p['Ui'].T)
            + np.dot(df, layer.p['Uf'].T)
        )

    dX = layer.bc['dX']

    return dX    # To previous layer
