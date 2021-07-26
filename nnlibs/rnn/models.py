# EpyNN/nnlibs/rnn/models.py
# Local application/library specific imports
from nnlibs.commons.models import Layer
from nnlibs.commons.maths import tanh, sigmoid, xavier
from nnlibs.rnn.forward import rnn_forward
from nnlibs.rnn.backward import rnn_backward
from nnlibs.rnn.parameters import (
    rnn_compute_shapes,
    rnn_initialize_parameters,
    rnn_update_gradients,
    rnn_update_parameters
)


class RNN(Layer):
    """
    Definition of a RNN Layer prototype

    Attributes
    ----------
    initialization : function
        Function used for weight initialization.
    activation : dict
        Activation functions as key-value pairs for log purpose.
    activate : function
        Activation function.
    lrate : list
        Learning rate along epochs for RNN layer
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
                activate_input=tanh,
                initialization=xavier,
                binary=False):

        super().__init__()

        self.initialization = initialization

        self.activation = {
            'activate': activate.__name__,
            'activate_input': activate_input.__name__,
        }

        self.activate = activate
        self.activate_input = activate_input

        self.d['h'] = hidden_size

        self.binary = binary

        self.lrate = []

    def compute_shapes(self, A):
        rnn_compute_shapes(self, A)
        return None

    def initialize_parameters(self):
        rnn_initialize_parameters(self)
        return None

    def forward(self, A):
        # Forward pass
        self.compute_shapes(A)
        A = rnn_forward(self, A)
        self.update_shapes(mode='forward')
        return A

    def backward(self, dA):
        # Backward pass
        dA = rnn_backward(self, dA)
        self.update_shapes(mode='backward')
        return dA

    def update_gradients(self):
        # Backward pass
        rnn_update_gradients(self)
        return None

    def update_parameters(self):
        # Update parameters
        rnn_update_parameters(self)
        return None
