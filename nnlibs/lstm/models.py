# EpyNN/nnlibs/lstm/models.py
# Local application/library specific imports
from nnlibs.commons.models import Layer
from nnlibs.commons.maths import (
    tanh,
    sigmoid,
    orthogonal,
    clip_gradient,
)
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
                activate=tanh,
                activate_output=sigmoid,
                activate_candidate=tanh,
                activate_input=sigmoid,
                activate_forget=sigmoid,
                initialization=orthogonal,
                clip_gradients=True,
                se_hPars=None):

        super().__init__(se_hPars)

        self.initialization = initialization

        self.activation = {
            'activate': activate.__name__,
            'activate_input': activate_input.__name__,
            'activate_candidate': activate_candidate.__name__,
            'activate_forget': activate_forget.__name__,
            'activate_output': activate_output.__name__,
        }

        self.activate = activate
        self.activate_input = activate_input
        self.activate_candidate = activate_candidate
        self.activate_forget = activate_forget
        self.activate_output = activate_output

        self.d['h'] = hidden_size

        self.clip_gradients = clip_gradients

        return None

    def compute_shapes(self, A):
        """Is a wrapper for :func:`nlibs.lstm.parameters.lstm_compute_shapes()`.
        """
        lstm_compute_shapes(self, A)

        return None

    def initialize_parameters(self):
        """Is a wrapper for :func:`nlibs.lstm.parameters.lstm_initialize_parameters()`.
        """
        lstm_initialize_parameters(self)

        return None

    def forward(self, A):
        """Is a wrapper for :func:`nlibs.lstm.forward.lstm_forward()`.
        """
        self.compute_shapes(A)
        A = lstm_forward(self, A)
        self.update_shapes(self.fc, self.fs)

        return A

    def backward(self, dA):
        """Is a wrapper for :func:`nlibs.lstm.backward.lstm_backward()`.
        """
        dA = lstm_backward(self, dA)
        self.update_shapes(self.bc, self.bs)

        return dA

    def compute_gradients(self):
        """Is a wrapper for :func:`nlibs.lstm.parameters.lstm_compute_gradients()`.
        """
        lstm_compute_gradients(self)

        if self.clip_gradients:
            clip_gradient(self)

        return None

    def update_parameters(self):
        """Is a wrapper for :func:`nlibs.lstm.parameters.lstm_update_parameters()`.
        """
        lstm_update_parameters(self)

        return None
