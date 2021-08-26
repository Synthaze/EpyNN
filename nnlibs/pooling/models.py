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

    :param pool_size: Height and width for pooling window, defaults to `(2, 2)`.
    :type pool_size: int or tuple[int], optional

    :param strides: Height and width to shift the pooling window by, defaults to `None` which equals `pool_size`.
    :type strides: int or tuple[int], optional

    :param stride: Walking step for filters in pooling layer.
    :type stride: int, optional

    :param pool: Pooling activation of units, defaults to :func:`np.max`.
    :type stride: function, optional
    """

    def __init__(self,
                 pool_size=(2, 2),
                 strides=None,
                 pool=np.max):

        super().__init__()

        pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        strides = strides if isinstance(strides, tuple) else pool_size

        self.d['ph'], self.d['pw'] = pool_size
        self.d['sh'], self.d['sw'] = strides
        self.pool = pool

        self.activation = { 'pool': self.pool.__name__ }
        self.trainable = False

        return None

    def compute_shapes(self, A):
        """Wrapper for :func:`nnlibs.pooling.parameters.pooling_compute_shapes()`.

        :param A: Output of forward propagation from previous layer.
        :type A: :class:`numpy.ndarray`
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

        :param A: Output of forward propagation from previous layer.
        :type A: :class:`numpy.ndarray`

        :return: Output of forward propagation for current layer.
        :rtype: :class:`numpy.ndarray`
        """
        self.compute_shapes(A)
        A = pooling_forward(self, A)
        self.update_shapes(self.fc, self.fs)

        return A

    def backward(self, dX):
        """Wrapper for :func:`nnlibs.pooling.backward.pooling_backward()`.

        :param dX: Output of backward propagation from next layer.
        :type dX: :class:`numpy.ndarray`

        :return: Output of backward propagation for current layer.
        :rtype: :class:`numpy.ndarray`
        """
        dX = pooling_backward(self, dX)
        self.update_shapes(self.bc, self.bs)

        return dX

    def compute_gradients(self):
        """Wrapper for :func:`nnlibs.pooling.parameters.pooling_compute_gradients()`. Dummy method, there is no gradients to compute in layer.
        """
        pooling_compute_gradients(self)

        return None

    def update_parameters(self):
        """Wrapper for :func:`nnlibs.pooling.parameters.pooling_update_parameters()`. Dummy method, there is no parameters to update in layer.
        """
        if self.trainable:
            pooling_update_parameters(self)

        return None
