# EpyNN/nnlibs/gru/models.py
# Local application/library specific imports
from nnlibs.commons.models import Layer
from nnlibs.commons.maths import tanh, sigmoid, xavier
from nnlibs.gru.forward import gru_forward
from nnlibs.gru.backward import gru_backward
from nnlibs.gru.parameters import (
    gru_compute_shapes,
    gru_initialize_parameters,
    gru_compute_gradients,
    gru_update_parameters
)


class GRU(Layer):
    """
    Definition of a GRU layer prototype.

    :param hidden_size: Number of GRU cells in one GRU layer.
    :type nodes: int

    :param activate: Activation function for output of GRU cells.
    :type activate: function

    :param activate_hidden: Activation function for hidden state of GRU cells.
    :type activate_hidden: function

    :param activate_update: Activation function for update gate in GRU cells.
    :type activate_update: function

    :param activate_reset: Activation function for reset gate in GRU cells.
    :type activate_reset: function

    :param initialization: Weight initialization function for GRU layer.
    :type initialization: function

    :param binary: Set the GRU layer from many-to-many to many-to-one mode.
    :type binary: bool
    """

    def __init__(self,
                hidden_size=10,
                activate=sigmoid,
                activate_hidden=tanh,
                activate_update=sigmoid,
                activate_reset=sigmoid,
                initialization=xavier,
                binary=False):

        super().__init__()

        self.initialization = initialization

        self.activation = {
            'activate': activate.__name__,
            'activate_input': activate_hidden.__name__,
            'activate_update': activate_update.__name__,
            'activate_reset': activate_reset.__name__,
        }

        self.activate = activate
        self.activate_update = activate_update
        self.activate_reset = activate_reset
        self.activate_input = activate_hidden

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

    def compute_gradients(self):
        # Backward pass
        gru_compute_gradients(self)
        return None

    def update_parameters(self):
        # Update parameters
        gru_update_parameters(self)
        return None
