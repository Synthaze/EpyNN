# EpyNN/nnlibs/dense/models.py
# Local application/library specific imports
from nnlibs.commons.models import Layer
from nnlibs.commons.maths import (
    sigmoid,
    xavier,
    activation_tune,
)
from nnlibs.dense.forward import dense_forward
from nnlibs.dense.backward import dense_backward
from nnlibs.dense.parameters import (
    dense_compute_shapes,
    dense_initialize_parameters,
    dense_compute_gradients,
    dense_update_parameters
)


class Dense(Layer):
    """
    Definition of a dense layer prototype.

    :param nodes: Number of nodes for dense layer.
    :type nodes: int

    :param activate: Activation function for output of nodes.
    :type activate: function

    :param initialization: Weight initialization function for dense layer.
    :type initialization: function
    """

    def __init__(self,
                nodes=1,
                activate=sigmoid,
                initialization=xavier,
                se_hPars=None):

        super().__init__(se_hPars)

        self.initialization = initialization

        self.activation = { 'activate': activate.__name__ }

        self.activate = activate

        self.d['n'] = nodes

        return None

    def compute_shapes(self, A):
        """Wrapper for :func:`nnlibs.dense.parameters.dense_compute_shapes()`.
        """
        dense_compute_shapes(self, A)

        return None

    def initialize_parameters(self):
        """Wrapper for :func:`nnlibs.dense.parameters.dense_initialize_parameters()`.
        """
        dense_initialize_parameters(self)

        return None

    def forward(self, A):
        """Wrapper for :func:`nnlibs.dense.forward.dense_forward()`.
        """
        activation_tune(self.se_hPars)
        A = dense_forward(self, A)
        self.update_shapes(self.fc, self.fs)

        return A

    def backward(self, dA):
        """Wrapper for :func:`nnlibs.dense.backward.dense_backward()`.
        """
        activation_tune(self.se_hPars)
        dA = dense_backward(self, dA)
        self.update_shapes(self.bc, self.bs)

        return dA

    def compute_gradients(self):
        """Wrapper for :func:`nnlibs.dense.parameters.dense_compute_gradients()`.
        """
        dense_compute_gradients(self)

        return None

    def update_parameters(self):
        """Wrapper for :func:`nnlibs.dense.parameters.dense_update_parameters()`.
        """
        dense_update_parameters(self)

        return None
