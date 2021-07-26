# EpyNN/nnlibs/conv/models.py
# Local application/library specific imports
from nnlibs.commons.models import Layer
from nnlibs.commons.maths import relu, xavier
from nnlibs.convolution.forward import convolution_forward
from nnlibs.convolution.backward import convolution_backward
from nnlibs.convolution.parameters import (
    convolution_compute_shapes,
    convolution_initialize_parameters,
    convolution_update_gradients,
    convolution_update_parameters
)


class Convolution(Layer):
    """
    Definition of a Convolution Layer prototype

    Attributes
    ----------
    initialization : function
        Function used for weight initialization.
    activation : dict
        Activation functions as key-value pairs for log purpose.
    activate : function
        Activation function.
    lrate : list
        Learning rate along epochs for Convolution layer

    Methods
    -------
    compute_shapes(A)
        .
    initialize_parameters()
        .
    forward(A)
        .
    backward(dA)
        .
    update_gradients()
        .
    update_parameters()
        .

    See Also
    --------
    nnlibs.commons.models.Layer :
        Layer Parent class which defines dictionary attributes for dimensions, parameters, gradients, shapes and caches. It also define the update_shapes() method.
    """

    def __init__(self,
                n_filters=1,
                f_width=3,
                depth=1,
                stride=1,
                padding=0,
                activate=relu,
                initialization=xavier):

        super().__init__()

        ### Init shapes
        self.n_filters = n_filters
        self.f_width = f_width
        self.depth = depth
        self.stride = stride
        self.padding = padding

        self.initialization = initialization

        self.activation = { 'activate': activate.__name__ }

        self.activate = activate

        self.d['n'] = n_filters
        self.d['w'] = f_width
        self.d['d'] = depth
        self.d['s'] = stride
        self.d['p'] = padding

        self.lrate = []

    def compute_shapes(self, A):
        convolution_compute_shapes(self, A)
        return None

    def initialize_parameters(self):
        convolution_initialize_parameters(self)
        return None

    def forward(self, A):
        # Forward pass
        self.compute_shapes(A)
        A = convolution_forward(self, A)
        #self.update_shapes(mode='forward')
        return A

    def backward(self, dA):
        # Backward pass
        dA = convolution_backward(self, dA)
        #self.update_shapes(mode='backward')
        return dA

    def update_gradients(self):
        # Backward pass
        convolution_update_gradients(self)
        return None

    def update_parameters(self):
        # Update parameters
        convolution_update_parameters(self)
        return None
