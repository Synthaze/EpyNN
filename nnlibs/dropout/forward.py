# EpyNN/nnlibs/dropout/forward.py
# Related third party imports
import numpy as np


def dropout_forward(layer, A):

    X = initialize_forward(layer, A)

    D = layer.np_rng.standard_normal(layer.fs['D'])

    D = layer.fc['D'] = (D < layer.d['k'])

    A = X * D

    A = A / layer.d['k']

    A = layer.fc['A'] = A

    return A


def initialize_forward(layer, A):

    X = layer.fc['X'] = A

    return X
