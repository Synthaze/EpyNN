# EpyNN/nnlibs/gru/models.py
# Local application/library specific imports
from nnlibs.commons.models import Layer
from nnlibs.commons.maths import (
    tanh,
    sigmoid,
    orthogonal,
    clip_gradient,
)
from nnlibs.gru.forward import gru_forward
from nnlibs.gru.backward import gru_backward
from nnlibs.gru.parameters import (
    gru_compute_shapes,
    gru_initialize_parameters,
    gru_compute_gradients,
    gru_update_parameters,
)


class GRU(Layer):
    """
    Definition of a GRU layer prototype.

    :param hidden_size: Number of GRU cells in one GRU layer.
    :type nodes: int

    :param activate: Activation function for output of GRU cells.
    :type activate: function

    :param activate_update: Activation function for update gate in GRU cells.
    :type activate_update: function

    :param activate_reset: Activation function for reset gate in GRU cells.
    :type activate_reset: function

    :param initialization: Weight initialization function for GRU layer.
    :type initialization: function
    """

    def __init__(self,
                hidden_size=10,
                activate=tanh,
                activate_update=sigmoid,
                activate_reset=sigmoid,
                initialization=orthogonal,
                clip_gradients=True):

        super().__init__()

        self.initialization = initialization

        self.activation = {
            'activate': activate.__name__,
            'activate_update': activate_update.__name__,
            'activate_reset': activate_reset.__name__,
        }

        self.activate = activate
        self.activate_update = activate_update
        self.activate_reset = activate_reset

        self.d['h'] = hidden_size

        self.clip_gradients = clip_gradients

        return None

    def compute_shapes(self, A):
        """Wrapper for :func:`nnlibs.gru.parameters.gru_compute_shapes()`.
        """
        gru_compute_shapes(self, A)

        return None

    def initialize_parameters(self):
        """Wrapper for :func:`nnlibs.gru.parameters.gru_initialize_parameters()`.
        """
        gru_initialize_parameters(self)

        return None

    def forward(self, A):
        """Wrapper for :func:`nnlibs.gru.forward.gru_forward()`.
        """
        self.compute_shapes(A)
        A = gru_forward(self, A)
        self.update_shapes(self.fc, self.fs)

        return A

    def backward(self, dA):
        """Wrapper for :func:`nnlibs.gru.backward.gru_backward()`.
        """
        dA = gru_backward(self, dA)
        self.update_shapes(self.bc, self.bs)

        return dA

    def compute_gradients(self):
        """Wrapper for :func:`nnlibs.gru.parameters.gru_compute_gradients()`.
        """
        gru_compute_gradients(self)

        if self.clip_gradients:
            clip_gradient(self)

        return None

    def update_parameters(self):
        """Wrapper for :func:`nnlibs.gru.parameters.gru_update_parameters()`.
        """
        gru_update_parameters(self)

        return None
