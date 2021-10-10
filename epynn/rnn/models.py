# EpyNN/epynn/rnn/models.py
# Local application/library specific imports
from epynn.commons.models import Layer
from epynn.commons.maths import (
    tanh,
    xavier,
    clip_gradient,
    activation_tune,
)
from epynn.rnn.forward import rnn_forward
from epynn.rnn.backward import rnn_backward
from epynn.rnn.parameters import (
    rnn_compute_shapes,
    rnn_initialize_parameters,
    rnn_compute_gradients,
    rnn_update_parameters
)


class RNN(Layer):
    """
    Definition of a RNN layer prototype.

    :param units: Number of unit cells in RNN layer, defaults to 1.
    :type units: int, optional

    :param activate: Non-linear activation of hidden state, defaults to `tanh`.
    :type activate: function, optional

    :param initialization: Weight initialization function for RNN layer, defaults to `xavier`.
    :type initialization: function, optional

    :param clip_gradients: May prevent exploding/vanishing gradients, defaults to `False`.
    :type clip_gradients: bool, optional

    :param sequences: Whether to return only the last hidden state or the full sequence, defaults to `False`.
    :type sequences: bool, optional

    :param se_hPars: Layer hyper-parameters, defaults to `None` and inherits from model.
    :type se_hPars: dict[str, str or float] or NoneType, optional
    """

    def __init__(self,
                 unit_cells=1,
                 activate=tanh,
                 initialization=xavier,
                 clip_gradients=True,
                 sequences=False,
                 se_hPars=None):
        """Initialize instance variable attributes.
        """
        super().__init__(se_hPars)

        self.d['u'] = unit_cells
        self.activate = activate
        self.initialization = initialization
        self.clip_gradients = clip_gradients
        self.sequences = sequences

        self.activation = { 'activate': self.activate.__name__ }
        self.trainable = True

        return None

    def compute_shapes(self, A):
        """Wrapper for :func:`epynn.rnn.parameters.rnn_compute_shapes()`.

        :param A: Output of forward propagation from previous layer.
        :type A: :class:`numpy.ndarray`
        """
        rnn_compute_shapes(self, A)

        return None

    def initialize_parameters(self):
        """Wrapper for :func:`epynn.rnn.parameters.rnn_initialize_parameters()`.
        """
        rnn_initialize_parameters(self)

        return None

    def forward(self, A):
        """Wrapper for :func:`epynn.rnn.forward.rnn_forward()`.

        :param A: Output of forward propagation from previous layer.
        :type A: :class:`numpy.ndarray`

        :return: Output of forward propagation for current layer.
        :rtype: :class:`numpy.ndarray`
        """
        self.compute_shapes(A)
        activation_tune(self.se_hPars)
        A = self.fc['A'] = rnn_forward(self, A)
        self.update_shapes(self.fc, self.fs)

        return A

    def backward(self, dX):
        """Wrapper for :func:`epynn.rnn.backward.rnn_backward()`.

        :param dX: Output of backward propagation from next layer.
        :type dX: :class:`numpy.ndarray`

        :return: Output of backward propagation for current layer.
        :rtype: :class:`numpy.ndarray`
        """
        activation_tune(self.se_hPars)
        dX = rnn_backward(self, dX)
        self.update_shapes(self.bc, self.bs)

        return dX

    def compute_gradients(self):
        """Wrapper for :func:`epynn.rnn.parameters.rnn_compute_gradients()`.
        """
        rnn_compute_gradients(self)

        if self.clip_gradients:
            clip_gradient(self)

        return None

    def update_parameters(self):
        """Wrapper for :func:`epynn.rnn.parameters.rnn_update_parameters()`.
        """
        if self.trainable:
            rnn_update_parameters(self)

        return None
