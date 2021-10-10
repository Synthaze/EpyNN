# EpyNN/epynn/gru/models.py
# Local application/library specific imports
from epynn.commons.models import Layer
from epynn.commons.maths import (
    tanh,
    sigmoid,
    orthogonal,
    clip_gradient,
    activation_tune,
)
from epynn.gru.forward import gru_forward
from epynn.gru.backward import gru_backward
from epynn.gru.parameters import (
    gru_compute_shapes,
    gru_initialize_parameters,
    gru_compute_gradients,
    gru_update_parameters,
)


class GRU(Layer):
    """
    Definition of a GRU layer prototype.

    :param units: Number of unit cells in GRU layer, defaults to 1.
    :type units: int, optional

    :param activate: Non-linear activation of hidden hat (hh) state, defaults to `tanh`.
    :type activate: function, optional

    :param activate_output: Non-linear activation of update gate, defaults to `sigmoid`.
    :type activate_output: function, optional

    :param activate_candidate: Non-linear activation of reset gate, defaults to `sigmoid`.
    :type activate_candidate: function, optional

    :param initialization: Weight initialization function for GRU layer, defaults to `orthogonal`.
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
                 activate_update=sigmoid,
                 activate_reset=sigmoid,
                 initialization=orthogonal,
                 clip_gradients=False,
                 sequences=False,
                 se_hPars=None):
        """Initialize instance variable attributes.
        """
        super().__init__(se_hPars)

        self.d['u'] = unit_cells
        self.activate = activate
        self.activate_update = activate_update
        self.activate_reset = activate_reset
        self.initialization = initialization
        self.clip_gradients = clip_gradients
        self.sequences = sequences

        self.activation = {
            'activate': self.activate.__name__,
            'activate_update': self.activate_update.__name__,
            'activate_reset': self.activate_reset.__name__,
        }
        self.trainable = True

        return None

    def compute_shapes(self, A):
        """Wrapper for :func:`epynn.gru.parameters.gru_compute_shapes()`.

        :param A: Output of forward propagation from previous layer.
        :type A: :class:`numpy.ndarray`
        """
        gru_compute_shapes(self, A)

        return None

    def initialize_parameters(self):
        """Wrapper for :func:`epynn.gru.parameters.gru_initialize_parameters()`.
        """
        gru_initialize_parameters(self)

        return None

    def forward(self, A):
        """Wrapper for :func:`epynn.gru.forward.gru_forward()`.

        :param A: Output of forward propagation from previous layer.
        :type A: :class:`numpy.ndarray`

        :return: Output of forward propagation for current layer.
        :rtype: :class:`numpy.ndarray`
        """
        self.compute_shapes(A)
        activation_tune(self.se_hPars)
        A = self.fc['A'] = gru_forward(self, A)
        self.update_shapes(self.fc, self.fs)

        return A

    def backward(self, dX):
        """Wrapper for :func:`epynn.gru.backward.gru_backward()`.

        :param dX: Output of backward propagation from next layer.
        :type dX: :class:`numpy.ndarray`

        :return: Output of backward propagation for current layer.
        :rtype: :class:`numpy.ndarray`
        """
        activation_tune(self.se_hPars)
        dX = gru_backward(self, dX)
        self.update_shapes(self.bc, self.bs)

        return dX

    def compute_gradients(self):
        """Wrapper for :func:`epynn.gru.parameters.gru_compute_gradients()`.
        """
        gru_compute_gradients(self)

        if self.clip_gradients:
            clip_gradient(self)

        return None

    def update_parameters(self):
        """Wrapper for :func:`epynn.gru.parameters.gru_update_parameters()`.
        """
        if self.trainable:
            gru_update_parameters(self)

        return None
