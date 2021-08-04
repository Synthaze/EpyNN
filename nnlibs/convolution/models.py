# EpyNN/nnlibs/conv/models.py
# Related third party imports
import numpy as np

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
                stride=1,
                padding=0,
                activate=np.abs,
                initialization=xavier):

        super().__init__()

        ### Init shapes
        self.initialization = initialization

        self.activation = { 'activate': activate.__name__ }

        self.activate = activate

        self.d['n'] = n_filters
        self.d['w'] = f_width
        self.d['s'] = stride
        self.d['p'] = padding

        return None

    def compute_shapes(self, A):
        """Wrapper for :func:`nnlibs.convolution.parameters.convolution_compute_shapes()`.
        """
        convolution_compute_shapes(self, A)

        return None

    def initialize_parameters(self):
        """Wrapper for :func:`nnlibs.convolution.parameters.convolution_initialize_parameters()`.
        """
        convolution_initialize_parameters(self)

        return None

    def forward(self, A):
        """Wrapper for :func:`nnlibs.convolution.forward.convolution_forward()`.
        """
        self.compute_shapes(A)
        A = convolution_forward(self, A)
        self.update_shapes(self.fc, self.fs)

        return A

    def backward(self, dA):
        """Wrapper for :func:`nnlibs.convolution.backward.convolution_backward()`.
        """
        dA = convolution_backward(self, dA)
        self.update_shapes(self.bc, self.bs)

        return dA

    def compute_gradients(self):
        """Wrapper for :func:`nnlibs.convolution.parameters.convolution_compute_gradients()`.
        """
        convolution_compute_gradients(self)

        return None

    def update_parameters(self):
        """Wrapper for :func:`nnlibs.convolution.parameters.convolution_update_parameters()`.
        """
        convolution_update_parameters(self)

        return None
