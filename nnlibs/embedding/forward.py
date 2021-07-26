# EpyNN/nnlibs/embedding/forward.py
# Related third party imports
import numpy as np


def embedding_forward(layer, A):

    X = initialize_forward(layer, A)

    A = layer.fc['A'] = X.T

    return A


def initialize_forward(layer, A):

    X = layer.fc['X'] = A

    return X
