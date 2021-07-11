#EpyNN/nnlibs/template/parameters.py
import nnlibs.commons.maths as cm


def set_activation(layer):

    args = layer.activation

    # Assign activation function to corresponding layer attribute

    return None


def init_shapes(layer):

    # Set layer shapes

    return None


def init_forward(layer,A):

    # Set and cache layer X and X.shape
    X = layer.fc['X'] = A
    layer.fs['X'] = X.shape

    return X


def init_params(layer):

    # Init parameters with corresponding function

    layer.init = False

    return None


def init_backward(layer,dA):

    # Cache dX (current) from dA (prev)
    dX = layer.bc['dX'] = dA

    return dX
