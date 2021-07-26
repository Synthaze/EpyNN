# EpyNN/nnlibs/lstm/models.py
# Local application/library specific imports
from nnlibs.commons.models import Layer
from nnlibs.commons.maths import tanh, sigmoid, xavier
from nnlibs.lstm.forward import lstm_forward
from nnlibs.lstm.backward import lstm_backward
from nnlibs.lstm.parameters import (
    lstm_compute_shapes,
    lstm_initialize_parameters,
    lstm_update_gradients,
    lstm_update_parameters
)


class LSTM(Layer):
    """
    Definition of a LSTM Layer prototype

    Attributes
    ----------
    initialization : function
        Function used for weight initialization.
    activation : dict
        Activation functions as key-value pairs for log purpose.
    activate : function
        Activation function.
    lrate : list
        Learning rate along epochs for LSTM layer
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
                activate_candidate=tanh,
                activate_forget=sigmoid,
                activate_output=sigmoid,
                activate_hidden=tanh,
                initialization=xavier,
                binary=False):

        super().__init__()

        self.initialization = initialization

        self.activation = {
            'activate': activate.__name__,
            'activate_input': activate_input.__name__,
            'activate_candidate': activate_candidate.__name__,
            'activate_forget': activate_forget.__name__,
            'activate_hidden': activate_hidden.__name__,
            'activate_output': activate_output.__name__,
        }

        self.activate = activate
        self.activate_input = activate_input
        self.activate_candidate = activate_candidate
        self.activate_forget = activate_forget
        self.activate_output = activate_output
        self.activate_hidden = activate_hidden

        self.d['h'] = hidden_size

        self.binary = binary

        self.lrate = []

    def compute_shapes(self, A):
        lstm_compute_shapes(self, A)
        return None

    def initialize_parameters(self):
        lstm_initialize_parameters(self)
        return None

    def forward(self, A):
        # Forward pass
        self.compute_shapes(A)
        A = lstm_forward(self, A)
        self.update_shapes(mode='forward')
        return A

    def backward(self, dA):
        # Backward pass
        dA = lstm_backward(self, dA)
        self.update_shapes(mode='backward')
        return dA

    def update_gradients(self):
        # Backward pass
        lstm_update_gradients(self)
        return None

    def update_parameters(self):
        # Update parameters
        lstm_update_parameters(self)
        return None
