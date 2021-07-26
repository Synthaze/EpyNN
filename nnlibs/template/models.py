# EpyNN/nnlibs/template/parameters.py
# Local application/library specific imports
from nnlibs.commons.models import Layer
from nnlibs.template.forward import template_forward
from nnlibs.template.backward import template_backward
from nnlibs.template.parameters import (
    template_compute_shapes,
    template_initialize_parameters,
    template_update_gradients,
    template_update_parameters
)


class Template(Layer):
    """
    Definition of a Template Layer prototype

    Attributes
    ----------
    initialization : function
        Function used for weight initialization.
    activation : dict
        Activation functions as key-value pairs for log purpose.
    activate : function
        Activation function.
    lrate : list
        Learning rate along epochs for Template layer

    Methods
    -------
    compute_shapes(A)
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

    def __init__(self,):

        super().__init__()

    def compute_shapes(self, A):
        template_compute_shapes(self, A)
        return None

    def initialize_parameters(self):
        template_initialize_parameters(self)
        return None

    def forward(self, A):
        # Forward pass
        self.compute_shapes(A)
        A = template_forward(self, A)
        self.update_shapes(mode='forward')
        return A

    def backward(self, dA):
        # Backward pass
        dA = template_backward(self, dA)
        self.update_shapes(mode='backward')
        return dA

    def update_gradients(self):
        # Backward pass
        template_update_gradients(self)
        return None

    def update_parameters(self):
        # Update parameters
        template_update_parameters(self)
        return None
