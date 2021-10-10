# EpyNN/epynn/convolution/models.py
# Local application/library specific imports
from epynn.commons.models import Layer
from epynn.commons.maths import (
    relu,
    xavier,
    activation_tune,
)
from epynn.convolution.forward import convolution_forward
from epynn.convolution.backward import convolution_backward
from epynn.convolution.parameters import (
    convolution_compute_shapes,
    convolution_initialize_parameters,
    convolution_compute_gradients,
    convolution_update_parameters,
)


class Convolution(Layer):
    """
    Definition of a convolution layer prototype.

    :param unit_filters: Number of unit filters in convolution layer, defaults to 1.
    :type unit_filters: int, optional

    :param filter_size: Height and width for convolution window, defaults to `(3, 3)`.
    :type filter_size: int or tuple[int], optional

    :param strides: Height and width to shift the convolution window by, defaults to `None` which equals `filter_size`.
    :type strides: int or tuple[int], optional

    :param padding: Number of zeros to pad each features plane with, defaults to 0.
    :type padding: int, optional

    :param activate: Non-linear activation of unit filters, defaults to `relu`.
    :type activate: function, optional

    :param initialization: Weight initialization function for convolution layer, defaults to `xavier`.
    :type initialization: function, optional

    :param use_bias: Whether the layer uses bias, defaults to `True`.
    :type use_bias: bool, optional

    :param se_hPars: Layer hyper-parameters, defaults to `None` and inherits from model.
    :type se_hPars: dict[str, str or float] or NoneType, optional
    """

    def __init__(self,
                 unit_filters=1,
                 filter_size=(3, 3),
                 strides=None,
                 padding=0,
                 activate=relu,
                 initialization=xavier,
                 use_bias=True,
                 se_hPars=None):
        """Initialize instance variable attributes.
        """
        super().__init__()

        filter_size = filter_size if isinstance(filter_size, tuple) else (filter_size, filter_size)
        strides = strides if isinstance(strides, tuple) else filter_size

        self.d['u'] = unit_filters
        self.d['fh'], self.d['fw'] = filter_size
        self.d['sh'], self.d['sw'] = strides
        self.d['p'] = padding
        self.activate = activate
        self.initialization = initialization
        self.use_bias = use_bias

        self.activation = { 'activate': activate.__name__ }
        self.trainable = True

        return None

    def compute_shapes(self, A):
        """Wrapper for :func:`epynn.convolution.parameters.convolution_compute_shapes()`.

        :param A: Output of forward propagation from previous layer.
        :type A: :class:`numpy.ndarray`
        """
        convolution_compute_shapes(self, A)

        return None

    def initialize_parameters(self):
        """Wrapper for :func:`epynn.convolution.parameters.convolution_initialize_parameters()`.
        """
        convolution_initialize_parameters(self)

        return None

    def forward(self, A):
        """Wrapper for :func:`epynn.convolution.forward.convolution_forward()`.

        :param A: Output of forward propagation from *previous* layer.
        :type A: :class:`numpy.ndarray`

        :return: Output of forward propagation for **current** layer.
        :rtype: :class:`numpy.ndarray`
        """
        activation_tune(self.se_hPars)
        A = convolution_forward(self, A)
        self.update_shapes(self.fc, self.fs)

        return A

    def backward(self, dX):
        """Wrapper for :func:`epynn.convolution.backward.convolution_backward()`.

        :param dX: Output of backward propagation from next layer.
        :type dX: :class:`numpy.ndarray`

        :return: Output of backward propagation for current layer.
        :rtype: :class:`numpy.ndarray`
        """
        activation_tune(self.se_hPars)
        dX = convolution_backward(self, dX)
        self.update_shapes(self.bc, self.bs)

        return dX

    def compute_gradients(self):
        """Wrapper for :func:`epynn.convolution.parameters.convolution_compute_gradients()`.
        """
        convolution_compute_gradients(self)

        return None

    def update_parameters(self):
        """Wrapper for :func:`epynn.convolution.parameters.convolution_update_parameters()`.
        """
        if self.trainable:
            convolution_update_parameters(self)

        return None
