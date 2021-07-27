# EpyNN/nnlibs/conv/models.py
# Local application/library specific imports
from nnlibs.commons.models import Layer
from nnlibs.commons.maths import relu, xavier
from nnlibs.convolution.forward import convolution_forward
from nnlibs.convolution.backward import convolution_backward
from nnlibs.convolution.parameters import (
    convolution_compute_shapes,
    convolution_initialize_parameters,
    convolution_compute_gradients,
    convolution_update_parameters
)


class Convolution(Layer):
    """
    Definition of a convolution layer prototype.

    :param n_filters: Number of filters in convolution layer.
    :type n_filters: int

    :param f_width: Filter width for filters in convolution layer.
    :type f_width: int

    :param activate: Activation function for convolution layer.
    :type activate: function

    :param depth: Depth or number of channels for input.
    :type depth: int

    :param stride: Walking step for filters.
    :type stride: int

    :param padding: ...
    :type padding: int

    :param initialization: Weight initialization function for convolution layer.
    :type initialization: bool
    """

    def __init__(self,
                n_filters=1,
                f_width=3,
                depth=1,
                stride=1,
                padding=0,
                activate=relu,
                initialization=xavier):

        super().__init__()

        ### Init shapes
        self.n_filters = n_filters
        self.f_width = f_width
        self.depth = depth
        self.stride = stride
        self.padding = padding

        self.initialization = initialization

        self.activation = { 'activate': activate.__name__ }

        self.activate = activate

        self.d['n'] = n_filters
        self.d['w'] = f_width
        self.d['d'] = depth
        self.d['s'] = stride
        self.d['p'] = padding

        self.lrate = []

    def compute_shapes(self, A):
        convolution_compute_shapes(self, A)
        return None

    def initialize_parameters(self):
        convolution_initialize_parameters(self)
        return None

    def forward(self, A):
        # Forward pass
        self.compute_shapes(A)
        A = convolution_forward(self, A)
        #self.update_shapes(mode='forward')
        return A

    def backward(self, dA):
        # Backward pass
        dA = convolution_backward(self, dA)
        #self.update_shapes(mode='backward')
        return dA

    def compute_gradients(self):
        # Backward pass
        convolution_compute_gradients(self)
        return None

    def update_parameters(self):
        # Update parameters
        convolution_update_parameters(self)
        return None
