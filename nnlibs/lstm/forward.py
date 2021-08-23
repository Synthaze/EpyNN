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

    cache_keys = ['h', 'hp', 'o', 'i', 'f', 'g', 'C', 'Cp']
    layer.fc.update({k: np.zeros(layer.fs['h']) for k in cache_keys})

    h = layer.fc['h'][:, 0]    # Hidden cell state
    C = layer.fc['C'][:, 0]    # Memory cell state

    return X, h, C


def lstm_forward(layer, A):
    """Forward propagate signal through LSTM cells to next layer.
    """
    # (1) Initialize cache, hidden and memory cell states
    X, h, C = initialize_forward(layer, A)

    # Iterate over sequence steps
    for s in range(layer.d['s']):

        # (2s) Slice sequence (m, s, v) with respect to step
        X = layer.fc['X'][:, s]

        # (3s) Retrieve previous cell states
        hp = layer.fc['hp'][:, s] = h    # Hidden
        Cp = layer.fc['Cp'][:, s] = C    # Memory

        # (4s) Activate forget gate
        f = np.dot(X, layer.p['Uf'])
        f += np.dot(hp, layer.p['Wf'])
        f += layer.p['bf']

        f = layer.fc['f'][:, s] = layer.activate_forget(f)

        # (5.1s) Activate input gate
        i = np.dot(X, layer.p['Ui'])
        i += np.dot(hp, layer.p['Wi'])
        i += layer.p['bi']

        i = layer.fc['i'][:, s] = layer.activate_input(i)

        # (5.2s) Activate candidate
        g = np.dot(X, layer.p['Ug'])
        g += np.dot(hp, layer.p['Wg'])
        g += layer.p['bg']

        g = layer.fc['g'][:, s] = layer.activate_candidate(g)

        # (6s) Compute current memory cell state
        C = layer.fc['C'][:, s] = Cp*f + i*g

        # (7s) Activate output gate
        o = np.dot(X, layer.p['Uo'])
        o += np.dot(hp, layer.p['Wo'])
        o += layer.p['bo']

        o = layer.fc['o'][:, s] = layer.activate_output(o)

        # (8s) Compute current hidden cell state
        h = layer.fc['h'][:, s] = o * layer.activate(C)

    # Return the last hidden cell state or the full sequence
    A = layer.fc['h'] if layer.sequences else layer.fc['h'][:, -1]

    return A    # To next layer
