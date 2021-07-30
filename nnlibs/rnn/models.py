# EpyNN/nnlibs/rnn/models.py
# Local application/library specific imports
from nnlibs.commons.models import Layer
from nnlibs.commons.maths import tanh, sigmoid, xavier, clip_gradient
from nnlibs.rnn.forward import rnn_forward
from nnlibs.rnn.backward import rnn_backward
from nnlibs.rnn.parameters import (
    rnn_compute_shapes,
    rnn_initialize_parameters,
    rnn_compute_gradients,
    rnn_update_parameters
)


class RNN(Layer):
    """
    Definition of a RNN layer prototype.

    :param hidden_size: Number of RNN cells in one RNN layer.
    :type hidden_size: int

    :param activate: Activation function for output of RNN cells.
    :type activate: function

    :param activate_hidden: Activation function for hidden state of RNN cells.
    :type activate_hidden: function

    :param initialization: Weight initialization function for RNN layer.
    :type initialization: function

    :param binary: Set the RNN layer from many-to-many to many-to-one mode.
    :type binary: bool
    """

    def __init__(self,
                hidden_size=10,
                activate=sigmoid,
                activate_hidden=tanh,
                initialization=xavier,
                binary=False):

        super().__init__()

        self.initialization = initialization

        self.activation = {
            'activate': activate.__name__,
            'activate_hidden': activate_hidden.__name__,
        }

        self.activate = activate
        self.activate_hidden = activate_hidden

        self.binary = binary

        self.d['h'] = hidden_size

        return None

    def compute_shapes(self, A):
        """Wrapper for ``nnlibs.rnn.parameters.rnn_compute_shapes()``.
        """
        rnn_compute_shapes(self, A)
        return None

    def initialize_parameters(self):
        """Wrapper for ``nnlibs.rnn.parameters.rnn_initialize_parameters()``.
        """
        rnn_initialize_parameters(self)

        return None

    def forward(self, A):
        """Wrapper for ``nnlibs.rnn.forward.rnn_forward()``.
        """
        self.compute_shapes(A)
        A = rnn_forward(self, A)
        self.update_shapes(self.fc, self.fs)

        return A

    def backward(self, dA):
        """Wrapper for ``nnlibs.rnn.backward.rnn_backward()``.
        """
        dA = rnn_backward(self, dA)
        self.update_shapes(self.bc, self.bs)

        return dA

    def compute_gradients(self):
        """Wrapper for ``nnlibs.rnn.parameters.rnn_compute_gradients()``.
        """
        rnn_compute_gradients(self)
        clip_gradient(self)

        return None

    def update_parameters(self):
        """Wrapper for ``nnlibs.rnn.parameters.rnn_update_parameters()``.
        """
        rnn_update_parameters(self)

        return None
