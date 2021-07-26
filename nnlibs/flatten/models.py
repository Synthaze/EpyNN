#EpyNN/nnlibs/flatten/models.py
# Local application/library specific imports
from nnlibs.commons.models import Layer
from nnlibs.flatten.forward import flatten_forward
from nnlibs.flatten.backward import flatten_backward
from nnlibs.flatten.parameters import (
    flatten_compute_shapes,
    flatten_initialize_parameters,
    flatten_update_gradients,
    flatten_update_parameters
)


class Flatten(Layer):
    """
    Definition of an Flatten Layer prototype

    Attributes
    ----------
    . : .
        .

    Methods
    -------
    compute_shapes()
        .
    initialize_parameters()
        .
    forward(A)
        .
    backward(dA)
        .
    update_gradients()
        .
    update_parameters()
        .

    See Also
    --------
    nnlibs.commons.models.Layer :
        Layer Parent class which defines dictionary attributes for dimensions, parameters, gradients, shapes and caches. It also define the update_shapes() method.
    """

    def __init__(self):

        super().__init__()


    def compute_shapes(self, A):
        flatten_compute_shapes(self, A)
        return None

    def initialize_parameters(self):
        flatten_initialize_parameters(self)
        return None

    def forward(self, A):
        # Forward pass
        self.compute_shapes(A)
        A = flatten_forward(self, A)
        self.update_shapes(mode='forward')
        return A

    def backward(self, dA):
        # Backward pass
        dA = flatten_backward(self, dA)
        self.update_shapes(mode='backward')
        return dA

    def update_gradients(self):
        flatten_update_gradients(self)
        return None

    def update_parameters(self):
        flatten_update_parameters(self)
        return None
