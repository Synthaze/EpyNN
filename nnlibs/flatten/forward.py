# EpyNN/nnlibs/flatten/forward.py
import numpy as np


def flatten_forward(layer,A):

    X = initialize_forward(layer, A)

    A = layer.fc['A'] = np.reshape(X, layer.fs['A'])

    return A


def initialize_forward(layer, A):

    X = layer.fc['X'] = A

    return X
