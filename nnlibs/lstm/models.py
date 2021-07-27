# EpyNN/nnlibs/lstm/models.py
# Local application/library specific imports
from nnlibs.commons.models import Layer
from nnlibs.commons.maths import tanh, sigmoid, xavier
from nnlibs.lstm.forward import lstm_forward
from nnlibs.lstm.backward import lstm_backward
from nnlibs.lstm.parameters import (
    lstm_compute_shapes,
    lstm_initialize_parameters,
    lstm_compute_gradients,
    lstm_update_parameters
)


class LSTM(Layer):
    """
    Definition of a LSTM layer prototype.

    :param hidden_size: Number of LSTM cells in one LSTM layer.
    :type nodes: int

    :param activate: Activation function for output of LSTM cells.
    :type activate: function

    :param activate_hidden: Activation function for hidden state of LSTM cells.
    :type activate_hidden: function

    :param activate_output: Activation function for output gate in LSTM cells.
    :type activate_output: function

    :param activate_candidate: Activation function for canidate gate in LSTM cells.
    :type activate_candidate: function

    :param activate_input: Activation function for input gate in LSTM cells.
    :type activate_input: function

    :param activate_forget: Activation function for forget gate in LSTM cells.
    :type activate_forget: function

    :param initialization: Weight initialization function for LSTM layer.
    :type initialization: function

    :param binary: Set the LSTM layer from many-to-many to many-to-one mode.
    :type binary: bool
    """

    def __init__(self,
                hidden_size=10,
                activate=sigmoid,
                activate_hidden=tanh,
                activate_output=sigmoid,
                activate_candidate=tanh,
                activate_input=sigmoid,
                activate_forget=sigmoid,
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

    def compute_gradients(self):
        # Backward pass
        lstm_compute_gradients(self)
        return None

    def update_parameters(self):
        # Update parameters
        lstm_update_parameters(self)
        return None
