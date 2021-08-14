# EpyNN/nnlibs/dropout/models.py
# Local application/library specific imports
from nnlibs.commons.models import Layer
from nnlibs.dropout.forward import dropout_forward
from nnlibs.dropout.backward import dropout_backward
from nnlibs.dropout.parameters import (
    dropout_compute_shapes,
    dropout_initialize_parameters,
    dropout_compute_gradients,
    dropout_update_parameters
)


class Dropout(Layer):
    """
    Definition of a dropout layer prototype.

    :param keep_prob: Probability to keep active one unit from previous layer.
    :type keep_prob: float
    """

    def __init__(self, keep_prob=0.5):

        super().__init__()

        self.d['k'] = keep_prob

        return None

    def compute_shapes(self, A):
        """Wrapper for :func:`nnlibs.dropout.parameters.dropout_compute_shapes()`.

        :param A: Output of forward propagation from previous layer.
        :type A: :class:`numpy.ndarray`
        """
        dropout_compute_shapes(self, A)

        return None

    def initialize_parameters(self):
        """Wrapper for :func:`nnlibs.dropout.parameters.dropout_initialize_parameters()`.
        """
        dropout_initialize_parameters(self)

        return None

    def forward(self, A):
        """Wrapper for :func:`nnlibs.dropout.forward.dropout_forward()`.
        """
        self.compute_shapes(A)
        A = dropout_forward(self, A)
        self.update_shapes(self.fc, self.fs)

        return A

    def backward(self, dA):
        """Wrapper for :func:`nnlibs.dropout.backward.dropout_backward()`.
        """
        dA = dropout_backward(self, dA)
        self.update_shapes(self.bc, self.bs)

        return dA

    def compute_gradients(self):
        """Wrapper for :func:`nnlibs.dropout.parameters.dropout_compute_gradients()`. Dummy method, there is no gradients to compute in layer.
        """
        dropout_compute_gradients(self)

        return None

    def update_parameters(self):
        """Wrapper for :func:`nnlibs.dropout.parameters.dropout_update_parameters()`. Dummy method, there is no parameters to update in layer.
        """
        dropout_update_parameters(self)

        return None
