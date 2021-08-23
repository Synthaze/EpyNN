# EpyNN/nnlibs/rnn/models.py
# Local application/library specific imports
from nnlibs.commons.models import Layer
from nnlibs.commons.maths import (
    tanh,
    xavier,
    clip_gradient,
    activation_tune,
)
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

    :param unit_cells: Number of RNN unit_cells in one RNN layer.
    :type unit_cells: int

    :param activate: Activation function for output of RNN unit_cells.
    :type activate: function

    :param initialization: Weight initialization function for RNN layer.
    :type initialization: function

    :param binary: Set the RNN layer from many-to-many to many-to-one mode.
    :type binary: bool
    """

    def __init__(self,
                unit_cells=1,
                activate=tanh,
                initialization=xavier,
                clip_gradients=True,
                sequences=False,
                se_hPars=None):

        super().__init__(se_hPars)

        self.d['u'] = unit_cells

        self.activate = activate

        self.activation = {
            'activate': activate.__name__,
        }

        self.initialization = initialization

        self.clip_gradients = clip_gradients

        self.sequences = sequences

        return None

    def compute_shapes(self, A):
        """Wrapper for :func:`nnlibs.rnn.parameters.rnn_compute_shapes()`.

        :param A: Output of forward propagation from previous layer.
        :type A: :class:`numpy.ndarray`
        """
        rnn_compute_shapes(self, A)

        return None

    def initialize_parameters(self):
        """Wrapper for :func:`nnlibs.rnn.parameters.rnn_initialize_parameters()`.
        """
        rnn_initialize_parameters(self)

        return None

    def forward(self, A):
        """Wrapper for :func:`nnlibs.rnn.forward.rnn_forward()`.
        """
        self.compute_shapes(A)
        activation_tune(self.se_hPars)
        A = rnn_forward(self, A)
        self.update_shapes(self.fc, self.fs)

        return A

    def backward(self, dX):
        """Wrapper for :func:`nnlibs.rnn.backward.rnn_backward()`.
        """
        activation_tune(self.se_hPars)
        dX = rnn_backward(self, dX)
        self.update_shapes(self.bc, self.bs)

        return dX

    def compute_gradients(self):
        """Wrapper for :func:`nnlibs.rnn.parameters.rnn_compute_gradients()`.
        """
        rnn_compute_gradients(self)

        if self.clip_gradients:
            clip_gradient(self)

        return None

    def update_parameters(self):
        """Wrapper for :func:`nnlibs.rnn.parameters.rnn_update_parameters()`.
        """
        rnn_update_parameters(self)

        return None
