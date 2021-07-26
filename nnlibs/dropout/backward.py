# EpyNN/nnlibs/dropout/backward.py
# Related third party imports
import numpy as np


def dropout_backward(layer, dA):

    dX = initialize_backward(layer, dA)

    dA = dX * layer.fc['D']

    dA /= layer.d['k']

    dA = layer.bc['dA'] = dA

    return dA


def initialize_backward(layer, dA):

    dX = layer.bc['dX'] = dA

    return dX
