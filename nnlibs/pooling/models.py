# EpyNN/nnlibs/pool/models.py
# Related third party imports
import numpy as np

# Local application/library specific imports
from nnlibs.commons.models import Layer
from nnlibs.pooling.forward import pooling_forward
from nnlibs.pooling.backward import pooling_backward
from nnlibs.pooling.parameters import (
    pooling_compute_shapes,
    pooling_initialize_parameters,
    pooling_compute_gradients,
    pooling_update_parameters
)


class Pooling(Layer):
    """
    Definition of a pooling layer prototype.

    :param f_width: Filter width for filters in pooling layer.
    :type f_width: int

    :param stride: Walking step for filters in pooling layer.
    :type stride: int

    :param pool: Pooling function in pooling layer.
    :type stride: function
    """

    def __init__(self,
                pool_size=(2, 2),
                stride=2,
                pool=np.max):

        super().__init__()

        self.activation = { 'pool': np.max.__name__ }

        self.pool = pool

        self.d['w'] = pool_size[0]
        self.d['h'] = pool_size[1]
        self.d['s'] = stride

        return None

    def compute_shapes(self, A):
        """Wrapper for :func:`nnlibs.pooling.parameters.pooling_compute_shapes()`.
        """
        pooling_compute_shapes(self, A)

        return None

    def initialize_parameters(self):
        """Wrapper for :func:`nnlibs.pooling.parameters.initialize_parameters()`.
        """
        pooling_initialize_parameters(self)

        return None

    def forward(self, A):
        """Wrapper for :func:`nnlibs.pooling.forward.pooling_forward()`.
        """
        self.compute_shapes(A)
        A = pooling_forward(self, A)
        self.update_shapes(self.fc, self.fs)

        return A

    def backward(self, dA):
        """Wrapper for :func:`nnlibs.pooling.backward.pooling_backward()`.
        """
        dA = pooling_backward(self, dA)
        self.update_shapes(self.bc, self.bs)

        return dA

    def compute_gradients(self):
        """Wrapper for :func:`nnlibs.pooling.parameters.pooling_compute_gradients()`.
        """
        pooling_compute_gradients(self)

        return None

    def update_parameters(self):
        """Wrapper for :func:`nnlibs.pooling.parameters.pooling_update_parameters()`.
        """
        pooling_update_parameters(self)

        return None
