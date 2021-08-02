# EpyNN/nnlibs/flatten/models.py
# Local application/library specific imports
from nnlibs.commons.models import Layer
from nnlibs.flatten.forward import flatten_forward
from nnlibs.flatten.backward import flatten_backward
from nnlibs.flatten.parameters import (
    flatten_compute_shapes,
    flatten_initialize_parameters,
    flatten_compute_gradients,
    flatten_update_parameters
)


class Flatten(Layer):
    """
    Definition of a flatten layer prototype.
    """

    def __init__(self):

        super().__init__()


    def compute_shapes(self, A):
        """Wrapper for :func:`nnlibs.flatten.parameters.flatten_compute_shapes()`.
        """
        flatten_compute_shapes(self, A)

        return None

    def initialize_parameters(self):
        """Wrapper for :func:`nnlibs.flatten.parameters.flatten_initialize_parameters()`.
        """
        flatten_initialize_parameters(self)

        return None

    def forward(self, A):
        """Wrapper for :func:`nnlibs.flatten.forward.flatten_forward()`.
        """
        self.compute_shapes(A)
        A = flatten_forward(self, A)
        self.update_shapes(self.fc, self.fs)

        return A

    def backward(self, dA):
        """Wrapper for :func:`nnlibs.flatten.backward.flatten_backward()`.
        """
        dA = flatten_backward(self, dA)
        self.update_shapes(self.bc, self.bs)

        return dA

    def compute_gradients(self):
        """Wrapper for :func:`nnlibs.flatten.parameters.flatten_compute_gradients()`.
        """
        flatten_compute_gradients(self)

        return None

    def update_parameters(self):
        """Wrapper for :func:`nnlibs.flatten.parameters.flatten_update_parameters()`.
        """
        flatten_update_parameters(self)

        return None
