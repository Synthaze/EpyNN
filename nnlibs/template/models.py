# EpyNN/nnlibs/template/parameters.py
# Local application/library specific imports
from nnlibs.commons.models import Layer
from nnlibs.template.forward import template_forward
from nnlibs.template.backward import template_backward
from nnlibs.template.parameters import (
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
        """Initialize instance variable attributes. Extended with ``super().__init__()`` which calls ``nnlibs.commons.models.Layer.__init__()`` defined in the parent class.

        """

        super().__init__()

        return None

    def compute_shapes(self, A):
        """Compute **shapes** and set dependent **dimensions**. Is a wrapper for ``nnlibs.template.parameters.template_compute_shapes()``.

        :param A: Output of forward propagation from *previous* layer.
        :type A: :class:`numpy.ndarray`
        """

        template_compute_shapes(self, A)

        return None

    def initialize_parameters(self):
        """Initialize **weight** and **bias** parameters from shapes. Is a wrapper for ``nnlibs.template.parameters.template_initialize_parameters()``.
        """

        template_initialize_parameters(self)

        return None

    def forward(self, A):
        """Forward propagation of signal through **current** layer. Is a wrapper for ``nnlibs.template.forward.template_forward()``.

        :param A: Output of forward propagation from *previous* layer.
        :type A: :class:`numpy.ndarray`

        :return: Output from **current** layer.
        :rtype: :class:`numpy.ndarray`
        """

        self.compute_shapes(A)
        A = template_forward(self, A)
        self.update_shapes(self.fc, self.fs)

        return A

    def backward(self, dA):
        """Backward propagation of error through **current** layer. Is a wrapper for ``nnlibs.template.backward.template_backward()``.

        :param dA: Output of backward propagation from *next* layer.
        :type dA: :class:`numpy.ndarray`

        :return: Output of backward propagation from **current** layer.
        :rtype: :class:`numpy.ndarray`
        """

        dA = template_backward(self, dA)
        self.update_shapes(self.bc, self.bs)

        return dA

    def compute_gradients(self):
        """Compute **gradients** with respect to **weight** and **bias** parameters for current layer. Is a wrapper for ``nnlibs.template.parameters.template_compute_gradients()``.
        """

        template_compute_gradients(self)

        return None

    def update_parameters(self):
        """Update **weight** and **bias** parameters with respect to **gradients** for current layer. Is a wrapper for ``nnlibs.template.parameters.template_update_parameters()``.
        """

        template_update_parameters(self)

        return None
