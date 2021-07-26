# EpyNN/nnlibs/gru/models.py
# Local application/library specific imports
from nnlibs.commons.models import Layer
from nnlibs.commons.maths import tanh, sigmoid, xavier
from nnlibs.gru.forward import gru_forward
from nnlibs.gru.backward import gru_backward
from nnlibs.gru.parameters import (
    gru_compute_shapes,
    gru_initialize_parameters,
    gru_update_gradients,
    gru_update_parameters
)


class GRU(Layer):
    """
    Definition of a GRU Layer prototype

    Attributes
    ----------
    initialization : function
        Function used for weight initialization.
    activation : dict
        Activation functions as key-value pairs for log purpose.
    activate : function
        Activation function.
    lrate : list
        Learning rate along epochs for GRU layer
    binary : bool
        .

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

    def __init__(self,
                hidden_size=10,
                activate=sigmoid,
                activate_update=sigmoid,
                activate_reset=sigmoid,
                activate_input=tanh,
                initialization=xavier,
                binary=False):

        super().__init__()

        self.initialization = initialization

        self.activation = {
            'activate': activate.__name__,
            'activate_update': activate_update.__name__,
            'activate_reset': activate_reset.__name__,
            'activate_input': activate_input.__name__,
        }

        self.activate = activate
        self.activate_update = activate_update
        self.activate_reset = activate_reset
        self.activate_input = activate_input

        self.d['h'] = hidden_size

        self.binary = binary

        self.lrate = []

    def compute_shapes(self, A):
        gru_compute_shapes(self, A)
        return None

    def initialize_parameters(self):
        gru_initialize_parameters(self)
        return None

    def forward(self, A):
        # Forward pass
        self.compute_shapes(A)
        A = gru_forward(self, A)
        self.update_shapes(mode='forward')
        return A

    def backward(self, dA):
        # Backward pass
        dA = gru_backward(self, dA)
        self.update_shapes(mode='backward')
        return dA

    def update_gradients(self):
        # Backward pass
        gru_update_gradients(self)
        return None

    def update_parameters(self):
        # Update parameters
        gru_update_parameters(self)
        return None
