# EpyNN/nnlibs/lstm/forward.py
# Related third party imports
import numpy as np


def initialize_forward(layer, A):
    """Forward cache initialization.

    :param layer: An instance of LSTM layer.
    :type layer: :class:`nnlibs.lstm.models.LSTM`

    :param A: Output of forward propagation from previous layer.
    :type A: :class:`numpy.ndarray`

    :return: Input of forward propagation for current layer.
    :rtype: :class:`numpy.ndarray`

    :return: Previous cell state initialized with zeros.
    :rtype: :class:`numpy.ndarray`

    :return: Previous memory state initialized with zeros.
    :rtype: :class:`numpy.ndarray`
    """
    X = layer.fc['X'] = A

    cache_keys = ['h', 'hp', 'o_', 'o', 'i_', 'i', 'f_', 'f', 'g_', 'g', 'C_', 'Cp_', 'C']
    layer.fc.update({k: np.zeros(layer.fs['h']) for k in cache_keys})

    h = layer.fc['h'][:, 0]    # Hidden cell state
    C_ = layer.fc['C_'][:, 0]  # Memory cell state

    return X, h, C_


def lstm_forward(layer, A):
    """Forward propagate signal to next layer.
    """
    # (1) Initialize cache, hidden and memory cell states
    X, h, C_ = initialize_forward(layer, A)

    # Iterate over sequence steps
    for s in range(layer.d['s']):

        # (2s) Slice sequence (m, s, v) with respect to step
        X = layer.fc['X'][:, s]

        # (3s) Retrieve previous cell states
        hp = layer.fc['hp'][:, s] = h    # Hidden
        Cp_ = layer.fc['Cp_'][:, s] = C_    # Memory

        # (4s) Activate forget gate
        f_ = layer.fc['f_'][:, s] = (
            np.dot(X, layer.p['Uf'])
            + np.dot(hp, layer.p['Vf'])
            + layer.p['bf']
        )

        f = layer.fc['f'][:, s] = layer.activate_forget(f_)

        # (5.1s) Activate input gate
        i_ = layer.fc['i_'][:, s] = (
            np.dot(X, layer.p['Ui'])
            + np.dot(hp, layer.p['Vi'])
            + layer.p['bi']
        )

        i = layer.fc['i'][:, s] = layer.activate_input(i_)

        # (5.2s) Activate candidate
        g_ = layer.fc['g_'][:, s] = (
            np.dot(X, layer.p['Ug'])
            + np.dot(hp, layer.p['Vg'])
            + layer.p['bg']
        )

        g = layer.fc['g'][:, s] = layer.activate_candidate(g_)

        # (6s) Activate output gate
        o_ = layer.fc['o_'][:, s] = (
            np.dot(X, layer.p['Uo'])
            + np.dot(hp, layer.p['Vo'])
            + layer.p['bo']
        )

        o = layer.fc['o'][:, s] = layer.activate_output(o_)

        # (7s) Compute current memory cell state
        C_ = layer.fc['C_'][:, s] = (
            Cp_ * f
            + i * g
        )

        C = layer.fc['C'][:, s] = layer.activate(C_)

        # (8s) Compute current hidden cell state
        h = layer.fc['h'][:, s] = o * C

    # Return the last hidden cell state or the full sequence
    A = layer.fc['h'] if layer.sequences else layer.fc['h'][:, -1]

    return A    # To next layer
