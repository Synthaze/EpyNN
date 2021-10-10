# EpyNN/epynn/dense/models.py
# Local application/library specific imports
from epynn.commons.models import Layer
from epynn.commons.maths import (
    sigmoid,
    xavier,
    activation_tune,
)
from epynn.dense.forward import dense_forward
from epynn.dense.backward import dense_backward
from epynn.dense.parameters import (
    dense_compute_shapes,
    dense_initialize_parameters,
    dense_compute_gradients,
    dense_update_parameters
)


class Dense(Layer):
    """
    Definition of a dense layer prototype.

    :param units: Number of units in dense layer, defaults to 1.
    :type units: int, optional

    :param activate: Non-linear activation of units, defaults to `sigmoid`.
    :type activate: function, optional

    :param initialization: Weight initialization function for dense layer, defaults to `xavier`.
    :type initialization: function, optional

    :param se_hPars: Layer hyper-parameters, defaults to `None` and inherits from model.
    :type se_hPars: dict[str, str or float] or NoneType, optional
    """

    def __init__(self,
                 units=1,
                 activate=sigmoid,
                 initialization=xavier,
                 se_hPars=None):
        """Initialize instance variable attributes.
        """
        super().__init__(se_hPars)

        self.d['u'] = units
        self.activate = activate
        self.initialization = initialization

        self.activation = { 'activate': self.activate.__name__ }
        self.trainable = True

        return None

    def compute_shapes(self, A):
        """Wrapper for :func:`epynn.dense.parameters.dense_compute_shapes()`.

        :param A: Output of forward propagation from previous layer.
        :type A: :class:`numpy.ndarray`
        """
        dense_compute_shapes(self, A)

        return None

    def initialize_parameters(self):
        """Wrapper for :func:`epynn.dense.parameters.dense_initialize_parameters()`.
        """
        dense_initialize_parameters(self)

        return None

    def forward(self, A):
        """Wrapper for :func:`epynn.dense.forward.dense_forward()`.

        :param A: Output of forward propagation from previous layer.
        :type A: :class:`numpy.ndarray`

        :return: Output of forward propagation for current layer.
        :rtype: :class:`numpy.ndarray`
        """
        activation_tune(self.se_hPars)
        A = dense_forward(self, A)
        self.update_shapes(self.fc, self.fs)

        return A

    def backward(self, dX):
        """Wrapper for :func:`epynn.dense.backward.dense_backward()`.

        :param dX: Output of backward propagation from next layer.
        :type dX: :class:`numpy.ndarray`

        :return: Output of backward propagation for current layer.
        :rtype: :class:`numpy.ndarray`
        """
        activation_tune(self.se_hPars)
        dX = dense_backward(self, dX)
        self.update_shapes(self.bc, self.bs)

        return dX

    def compute_gradients(self):
        """Wrapper for :func:`epynn.dense.parameters.dense_compute_gradients()`.
        """
        dense_compute_gradients(self)

        return None

    def update_parameters(self):
        """Wrapper for :func:`epynn.dense.parameters.dense_update_parameters()`.
        """
        if self.trainable:
            dense_update_parameters(self)

        return None
