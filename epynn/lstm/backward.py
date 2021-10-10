# EpyNN/epynn/lstm/backward.py
# Related third party imports
import numpy as np


def initialize_backward(layer, dX):
    """Backward cache initialization.

    :param layer: An instance of LSTM layer.
    :type layer: :class:`epynn.lstm.models.LSTM`

    :param dX: Output of backward propagation from next layer.
    :type dX: :class:`numpy.ndarray`

    :return: Input of backward propagation for current layer.
    :rtype: :class:`numpy.ndarray`

    :return: Next hidden state initialized with zeros.
    :rtype: :class:`numpy.ndarray`

    :return: Next memory state initialized with zeros.
    :rtype: :class:`numpy.ndarray`
    """
    if layer.sequences:
        dA = dX                         # Full length sequence
    elif not layer.sequences:
        dA = np.zeros(layer.fs['h'])    # Empty full length sequence
        dA[:, -1] = dX                  # Assign to last index

    cache_keys = [
        'dh_', 'dh', 'dhn',
        'dC_', 'dC', 'dCn',
        'do_', 'dg_', 'di_', 'df_', 'dz_'
    ]

    layer.bc.update({k: np.zeros(layer.fs['h']) for k in cache_keys})

    layer.bc['dA'] = dA
    layer.bc['dX'] = np.zeros(layer.fs['X'])    # To previous layer

    dh = layer.bc['dh'][:, 0]                   # To previous step
    dC = layer.bc['dC'][:, 0]                   # To previous step

    return dA, dh, dC


def lstm_backward(layer, dX):
    """Backward propagate error gradients to previous layer.
    """
    # (1) Initialize cache, hidden and memory state gradients
    dA, dh, dC = initialize_backward(layer, dX)

    # Reverse iteration over sequence steps
    for s in reversed(range(layer.d['s'])):

        # (2s) Slice sequence (m, s, u) w.r.t step
        dA = layer.bc['dA'][:, s]          # dL/dA

        # (3s) Gradient of the loss w.r.t. next states
        dhn = layer.bc['dhn'][:, s] = dh   # (3.1) dL/dhn
        dCn = layer.bc['dCn'][:, s] = dC   # (3.2) dL/dCn

        # (4s) Gradient of the loss w.r.t hidden state h_
        dh_ = layer.bc['dh_'][:, s] = (
            (dA + dhn)
        )   # dL/dh_

        # (5s) Gradient of the loss w.r.t memory state C_
        dC_ = layer.bc['dC_'][:, s] = (
            dh_
            * layer.fc['o'][:, s]
            * layer.activate(layer.fc['C_'][:, s], deriv=True)
            + dCn
        )   # dL/dC_

        # (6s) Gradient of the loss w.r.t output gate o_
        do_ = layer.bc['do_'][:, s] = (
            dh_
            * layer.fc['C'][:, s]
            * layer.activate_output(layer.fc['o_'][:, s], deriv=True)
        )   # dL/do_

        # (7s) Gradient of the loss w.r.t candidate g_
        dg_ = layer.bc['dg_'][:, s] = (
            dC_
            * layer.fc['i'][:, s]
            * layer.activate_candidate(layer.fc['g_'][:, s], deriv=True)
        )   # dL/dg_

        # (8s) Gradient of the loss w.r.t input gate i_
        di_ = layer.bc['di_'][:, s] = (
            dC_
            * layer.fc['g'][:, s]
            * layer.activate_input(layer.fc['i_'][:, s], deriv=True)
        )   # dL/di_

        # (9s) Gradient of the loss w.r.t forget gate f_
        df_ = layer.bc['df_'][:, s] = (
            dC_
            * layer.fc['Cp_'][:, s]
            * layer.activate_forget(layer.fc['f_'][:, s], deriv=True)
        )   # dL/df_

        # (10s) Gradient of the loss w.r.t memory state C
        dC = layer.bc['dC'][:, s] = (
            dC_
            * layer.fc['f'][:, s]
        )   # dL/dC

        # (11s) Gradient of the loss w.r.t hidden state h
        dh = layer.bc['dh'][:, s] = (
            np.dot(do_, layer.p['Vo'].T)
            + np.dot(dg_, layer.p['Vg'].T)
            + np.dot(di_, layer.p['Vi'].T)
            + np.dot(df_, layer.p['Vf'].T)
        )   # dL/dh

        # (12s) Gradient of the loss w.r.t hidden state X
        dX = layer.bc['dX'][:, s] = (
            np.dot(dg_, layer.p['Ug'].T)
            + np.dot(do_, layer.p['Uo'].T)
            + np.dot(di_, layer.p['Ui'].T)
            + np.dot(df_, layer.p['Uf'].T)
        )   # dL/dX

    dX = layer.bc['dX']

    return dX    # To previous layer
