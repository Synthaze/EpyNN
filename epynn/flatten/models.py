# EpyNN/epynn/flatten/models.py
# Local application/library specific imports
from epynn.commons.models import Layer
from epynn.flatten.forward import flatten_forward
from epynn.flatten.backward import flatten_backward
from epynn.flatten.parameters import (
    flatten_compute_shapes,
    flatten_initialize_parameters,
    flatten_compute_gradients,
    flatten_update_parameters
)


class Flatten(Layer):
    """
    Definition of a flatten layer prototype.
    """

    def __init__(self):
        """Initialize instance variable attributes.
        """
        super().__init__()

        self.trainable = False

        return None

    def compute_shapes(self, A):
        """Wrapper for :func:`epynn.flatten.parameters.flatten_compute_shapes()`.

        :param A: Output of forward propagation from previous layer.
        :type A: :class:`numpy.ndarray`
        """
        flatten_compute_shapes(self, A)

        return None

    def initialize_parameters(self):
        """Wrapper for :func:`epynn.flatten.parameters.flatten_initialize_parameters()`.
        """
        flatten_initialize_parameters(self)

        return None

    def forward(self, A):
        """Wrapper for :func:`epynn.flatten.forward.flatten_forward()`.

        :param A: Output of forward propagation from previous layer.
        :type A: :class:`numpy.ndarray`

        :return: Output of forward propagation for current layer.
        :rtype: :class:`numpy.ndarray`
        """
        self.compute_shapes(A)
        A = flatten_forward(self, A)
        self.update_shapes(self.fc, self.fs)

        return A

    def backward(self, dX):
        """Wrapper for :func:`epynn.flatten.backward.flatten_backward()`.

        :param dX: Output of backward propagation from next layer.
        :type dX: :class:`numpy.ndarray`

        :return: Output of backward propagation for current layer.
        :rtype: :class:`numpy.ndarray`
        """
        dX = flatten_backward(self, dX)
        self.update_shapes(self.bc, self.bs)

        return dX

    def compute_gradients(self):
        """Wrapper for :func:`epynn.flatten.parameters.flatten_compute_gradients()`. Dummy method, there are no gradients to compute in layer.
        """
        flatten_compute_gradients(self)

        return None

    def update_parameters(self):
        """Wrapper for :func:`epynn.flatten.parameters.flatten_update_parameters()`. Dummy method, there are no parameters to update in layer.
        """
        if self.trainable:
            flatten_update_parameters(self)

        return None
