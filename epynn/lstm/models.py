# EpyNN/epynn/lstm/models.py
# Local application/library specific imports
from epynn.commons.models import Layer
from epynn.commons.maths import (
    tanh,
    sigmoid,
    orthogonal,
    clip_gradient,
    activation_tune,
)
from epynn.lstm.forward import lstm_forward
from epynn.lstm.backward import lstm_backward
from epynn.lstm.parameters import (
    lstm_compute_shapes,
    lstm_initialize_parameters,
    lstm_compute_gradients,
    lstm_update_parameters
)


class LSTM(Layer):
    """
    Definition of a LSTM layer prototype.

    :param units: Number of unit cells in LSTM layer, defaults to 1.
    :type units: int, optional

    :param activate: Non-linear activation of hidden and memory states, defaults to `tanh`.
    :type activate: function, optional

    :param activate_output: Non-linear activation of output gate, defaults to `sigmoid`.
    :type activate_output: function, optional

    :param activate_candidate: Non-linear activation of candidate, defaults to `tanh`.
    :type activate_candidate: function, optional

    :param activate_input: Non-linear activation of input gate, defaults to `sigmoid`.
    :type activate_input: function, optional

    :param activate_forget: Non-linear activation of forget gate, defaults to `sigmoid`.
    :type activate_forget: function, optional

    :param initialization: Weight initialization function for LSTM layer, defaults to `orthogonal`.
    :type initialization: function, optional

    :param clip_gradients: May prevent exploding/vanishing gradients, defaults to `False`.
    :type clip_gradients: bool, optional

    :param sequences: Whether to return only the last hidden state or the full sequence, defaults to `False`.
    :type sequences: bool, optional

    :param se_hPars: Layer hyper-parameters, defaults to `None` and inherits from model.
    :type se_hPars: dict[str, str or float] or NoneType, optional
    """

    def __init__(self,
                 unit_cells=1,
                 activate=tanh,
                 activate_output=sigmoid,
                 activate_candidate=tanh,
                 activate_input=sigmoid,
                 activate_forget=sigmoid,
                 initialization=orthogonal,
                 clip_gradients=False,
                 sequences=False,
                 se_hPars=None):
        """Initialize instance variable attributes.
        """
        super().__init__(se_hPars)

        self.d['u'] = unit_cells
        self.activate = activate
        self.activate_output = activate_output
        self.activate_candidate = activate_candidate
        self.activate_input = activate_input
        self.activate_forget = activate_forget
        self.initialization = initialization
        self.clip_gradients = clip_gradients
        self.sequences = sequences

        self.activation = {
            'activate': self.activate.__name__,
            'activate_output': self.activate_output.__name__,
            'activate_candidate': self.activate_candidate.__name__,
            'activate_input': self.activate_input.__name__,
            'activate_forget': self.activate_forget.__name__,
        }
        self.trainable = True

        return None

    def compute_shapes(self, A):
        """Is a wrapper for :func:`epynn.lstm.parameters.lstm_compute_shapes()`.

        :param A: Output of forward propagation from previous layer.
        :type A: :class:`numpy.ndarray`
        """
        lstm_compute_shapes(self, A)

        return None

    def initialize_parameters(self):
        """Is a wrapper for :func:`epynn.lstm.parameters.lstm_initialize_parameters()`.
        """
        lstm_initialize_parameters(self)

        return None

    def forward(self, A):
        """Is a wrapper for :func:`epynn.lstm.forward.lstm_forward()`.

        :param A: Output of forward propagation from previous layer.
        :type A: :class:`numpy.ndarray`

        :return: Output of forward propagation for current layer.
        :rtype: :class:`numpy.ndarray`
        """
        self.compute_shapes(A)
        activation_tune(self.se_hPars)
        A = self.fc['A'] = lstm_forward(self, A)
        self.update_shapes(self.fc, self.fs)

        return A

    def backward(self, dX):
        """Is a wrapper for :func:`epynn.lstm.backward.lstm_backward()`.

        :param dX: Output of backward propagation from next layer.
        :type dX: :class:`numpy.ndarray`

        :return: Output of backward propagation for current layer.
        :rtype: :class:`numpy.ndarray`
        """
        activation_tune(self.se_hPars)
        dX = lstm_backward(self, dX)
        self.update_shapes(self.bc, self.bs)

        return dX

    def compute_gradients(self):
        """Is a wrapper for :func:`epynn.lstm.parameters.lstm_compute_gradients()`.
        """
        lstm_compute_gradients(self)

        if self.clip_gradients:
            clip_gradient(self)

        return None

    def update_parameters(self):
        """Is a wrapper for :func:`epynn.lstm.parameters.lstm_update_parameters()`.
        """
        if self.trainable:
            lstm_update_parameters(self)

        return None
