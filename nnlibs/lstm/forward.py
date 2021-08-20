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

    cache_keys = ['h', 'o', 'i', 'f', 'g', 'C']

    layer.fc.update({k: np.zeros(layer.fs['h']) for k in cache_keys})

    hp = np.zeros_like(layer.fc['h'][:, 0])
    C = np.zeros_like(layer.fc['C'][:, 0])

    return X, hp, C


def lstm_forward(layer, A):
    """Forward propagate signal through LSTM cells to next layer.
    """
    # (1) Initialize cache, hidden and memory cell states
    X, hp, C = initialize_forward(layer, A)

    # Iterate over sequence steps
    for s in range(layer.d['s']):

        # (2s) Slice sequence (m, s, v) with respect to step
        X = layer.fc['X'][:, s]

        # (3s) Compute forget gate
        f = np.dot(X, layer.p['Uf'])
        f += np.dot(hp, layer.p['Wf'])
        f += layer.p['bf']

        f = layer.fc['f'][:, s] = layer.activate_forget(f)

        # (4.1s) Compute input gate
        i = np.dot(X, layer.p['Ui'])
        i += np.dot(hp, layer.p['Wi'])
        i += layer.p['bi']

        i = layer.fc['i'][:, s] = layer.activate_input(i)

        # (4.2s) Activate candidate
        g = np.dot(X, layer.p['Ug'])
        g += np.dot(hp, layer.p['Wg'])
        g += layer.p['bg']

        g = layer.fc['g'][:, s] = layer.activate_candidate(g)

        # (5s) Compute memory cell state
        C = layer.fc['C'][:, s] = C*f + i * g

        # (6s) Compute output gate
        o = np.dot(X, layer.p['Uo'])
        o += np.dot(hp, layer.p['Wo'])
        o += layer.p['bo']

        o = layer.fc['o'][:, s] = layer.activate_output(o)

        # (7s) Activate hidden cell state
        h = layer.fc['h'][:, s] = o * layer.activate(C)

    # Return all or only the last hidden cell state
    A = layer.fc['h'] if layer.sequences else layer.fc['h'][:, -1]

    layer.fc['A'] = A

    return A    # To next layer
