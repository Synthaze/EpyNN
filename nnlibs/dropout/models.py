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

    def compute_shapes(self, A):
        dropout_compute_shapes(self, A)
        return None

    def initialize_parameters(self):
        dropout_initialize_parameters(self)
        return None

    def forward(self, A):
        # Forward pass
        self.compute_shapes(A)
        A = dropout_forward(self, A)
        self.update_shapes(mode='forward')
        return A

    def backward(self, dA):
        # Backward pass
        dA = dropout_backward(self, dA)
        self.update_shapes(mode='backward')
        return dA

    def compute_gradients(self):
        # Backward pass
        dropout_compute_gradients(self)
        return None

    def update_parameters(self):
        # Update parameters
        dropout_update_parameters(self)
        return None
