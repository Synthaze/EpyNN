# EpyNN/epynn/dropout/models.py
# Local application/library specific imports
from epynn.commons.models import Layer
from epynn.dropout.forward import dropout_forward
from epynn.dropout.backward import dropout_backward
from epynn.dropout.parameters import (
    dropout_compute_shapes,
    dropout_initialize_parameters,
    dropout_compute_gradients,
    dropout_update_parameters
)


class Dropout(Layer):
    """
    Definition of a dropout layer prototype.

    :param drop_prob: Probability to drop one data point from previous layer to next layer, defaults to 0.5.
    :type drop_prob: float, optional

    :param axis: Compute and apply dropout mask along defined axis, defaults to all axis.
    :type axis: int or tuple[int], optional
    """

    def __init__(self, drop_prob=0.5, axis=()):
        """Initialize instance variable attributes.
        """
        super().__init__()

        axis = axis if isinstance(axis, tuple) else (axis,)

        self.d['d'] = drop_prob
        self.d['a'] = axis

        self.trainable = False

        return None

    def compute_shapes(self, A):
        """Wrapper for :func:`epynn.dropout.parameters.dropout_compute_shapes()`.

        :param A: Output of forward propagation from previous layer.
        :type A: :class:`numpy.ndarray`
        """
        dropout_compute_shapes(self, A)

        return None

    def initialize_parameters(self):
        """Wrapper for :func:`epynn.dropout.parameters.dropout_initialize_parameters()`.
        """
        dropout_initialize_parameters(self)

        return None

    def forward(self, A):
        """Wrapper for :func:`epynn.dropout.forward.dropout_forward()`.

        :param A: Output of forward propagation from previous layer.
        :type A: :class:`numpy.ndarray`

        :return: Output of forward propagation for current layer.
        :rtype: :class:`numpy.ndarray`
        """
        self.compute_shapes(A)
        A = self.fc['A'] = dropout_forward(self, A)
        self.update_shapes(self.fc, self.fs)

        return A

    def backward(self, dX):
        """Wrapper for :func:`epynn.dropout.backward.dropout_backward()`.

        :param dX: Output of backward propagation from next layer.
        :type dX: :class:`numpy.ndarray`

        :return: Output of backward propagation for current layer.
        :rtype: :class:`numpy.ndarray`
        """
        dX = dropout_backward(self, dX)
        self.update_shapes(self.bc, self.bs)

        return dX

    def compute_gradients(self):
        """Wrapper for :func:`epynn.dropout.parameters.dropout_compute_gradients()`. Dummy method, there are no gradients to compute in layer.
        """
        dropout_compute_gradients(self)

        return None

    def update_parameters(self):
        """Wrapper for :func:`epynn.dropout.parameters.dropout_update_parameters()`. Dummy method, there are no parameters to update in layer.
        """
        if self.trainable:
            dropout_update_parameters(self)

        return None
