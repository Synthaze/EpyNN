# EpyNN/epynn/template/parameters
# Local application/library specific imports
from epynn.commons.models import Layer
from epynn.commons.maths import activation_tune
from epynn.template.forward import template_forward
from epynn.template.backward import template_backward
from epynn.template.parameters import (
    template_compute_shapes,
    template_initialize_parameters,
    template_compute_gradients,
    template_update_parameters
)


class Template(Layer):
    """
    Definition of a template layer prototype. This is a pass-through or inactive layer prototype which contains method definitions used for all active layers. For all layer prototypes, methods are wrappers of functions which contain the specific implementations.
    """

    def __init__(self):
        """Initialize instance variable attributes. Extended with ``super().__init__()`` which calls :func:`epynn.commons.models.Layer.__init__()` defined in the parent class.

        :ivar trainable: Whether layer's parameters should be trainable.
        :vartype trainable: bool
        """
        super().__init__()

        self.trainable = True

        return None

    def compute_shapes(self, A):
        """Is a wrapper for :func:`epynn.template.parameters.template_compute_shapes()`.

        :param A: Output of forward propagation from *previous* layer.
        :type A: :class:`numpy.ndarray`
        """
        template_compute_shapes(self, A)

        return None

    def initialize_parameters(self):
        """Is a wrapper for :func:`epynn.template.parameters.template_initialize_parameters()`.
        """
        template_initialize_parameters(self)

        return None

    def forward(self, A):
        """Is a wrapper for :func:`epynn.template.forward.template_forward()`.

        :param A: Output of forward propagation from *previous* layer.
        :type A: :class:`numpy.ndarray`

        :return: Output of forward propagation for **current** layer.
        :rtype: :class:`numpy.ndarray`
        """
        self.compute_shapes(A)
        activation_tune(self.se_hPars)
        A = template_forward(self, A)
        self.update_shapes(self.fc, self.fs)

        return A

    def backward(self, dX):
        """Is a wrapper for :func:`epynn.template.backward.template_backward()`.

        :param dX: Output of backward propagation from *next* layer.
        :type dX: :class:`numpy.ndarray`

        :return: Output of backward propagation for **current** layer.
        :rtype: :class:`numpy.ndarray`
        """
        activation_tune(self.se_hPars)
        dX = template_backward(self, dX)
        self.update_shapes(self.bc, self.bs)

        return dX

    def compute_gradients(self):
        """Is a wrapper for :func:`epynn.template.parameters.template_compute_gradients()`. Dummy method, there are no gradients to compute in layer.
        """
        template_compute_gradients(self)

        return None

    def update_parameters(self):
        """Is a wrapper for :func:`epynn.template.parameters.template_update_parameters()`. Dummy method, there are no parameters to update in layer.
        """
        if self.trainable:
            template_update_parameters(self)

        return None
