# EpyNN/nnlibs/conv/models.py
# Related third party imports
import numpy as np

# Local application/library specific imports
from nnlibs.commons.models import Layer
from nnlibs.commons.maths import identity, xavier
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

    :param padding: Zeros to add on each side of each image.
    :type padding: int

    :param initialization: Weight initialization function for convolution layer.
    :type initialization: bool
    """

    def __init__(self,
                n_filters=1,
                filter_size=(3, 3),
                strides=None,
                padding=0,
                activate=identity,
                initialization=xavier,
                use_bias=True):

        super().__init__()

        filter_size = filter_size if isinstance(filter_size, tuple) else (filter_size, filter_size)

        self.d['n'] = n_filters
        self.d['fh'], self.d['fw'] = filter_size
        self.d['sh'], self.d['sw'] = strides if isinstance(strides, tuple) else filter_size

        self.d['p'] = padding

        self.activate = activate
        self.initialization = initialization
        self.use_bias = use_bias

        self.activation = { 'activate': activate.__name__ }
        self.trainable = True

        return None

    def compute_shapes(self, A):
        """Wrapper for :func:`nnlibs.convolution.parameters.convolution_compute_shapes()`.

        :param A: Output of forward propagation from previous layer.
        :type A: :class:`numpy.ndarray`
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

    def backward(self, dX):
        """Wrapper for :func:`nnlibs.convolution.backward.convolution_backward()`.
        """
        dX = convolution_backward(self, dX)
        self.update_shapes(self.bc, self.bs)

        return dX

    def compute_gradients(self):
        """Wrapper for :func:`nnlibs.convolution.parameters.convolution_compute_gradients()`.
        """
        convolution_compute_gradients(self)

        return None

    def update_parameters(self):
        """Wrapper for :func:`nnlibs.convolution.parameters.convolution_update_parameters()`.
        """
        if self.trainable:
            convolution_update_parameters(self)

        return None
