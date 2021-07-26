# EpyNN/nnlibs/dense/models.py
# Local application/library specific imports
from nnlibs.commons.models import Layer
from nnlibs.commons.maths import softmax, xavier
from nnlibs.dense.forward import dense_forward
from nnlibs.dense.backward import dense_backward
from nnlibs.dense.parameters import (
    dense_compute_shapes,
    dense_initialize_parameters,
    dense_update_gradients,
    dense_update_parameters
)


class Dense(Layer):
    """
    Definition of a Dense Layer prototype

    Attributes
    ----------
    initialization : function
        Function used for weight initialization.
    activation : dict
        Activation functions as key-value pairs for log purpose.
    activate : function
        Activation function.
    lrate : list
        Learning rate along epochs for Dense layer

    Methods
    -------
    compute_shapes()
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
                nodes=2,
                activate=softmax,
                initialization=xavier,
                ):

        super().__init__()

        self.initialization = initialization

        self.activation = { 'activate': activate.__name__ }

        self.activate = activate

        self.lrate = []

        # Store the number of nodes in the dimension dictionary
        self.d['n'] = nodes

    def compute_shapes(self, A):
        dense_compute_shapes(self, A)
        return None

    def initialize_parameters(self):
        dense_initialize_parameters(self)
        return None

    def forward(self, A):
        # Forward pass
        A = dense_forward(self, A)
        self.update_shapes(mode='forward')
        return A

    def backward(self, dA):
        # Backward pass
        dA = dense_backward(self, dA)
        self.update_shapes(mode='backward')
        return dA

    def update_gradients(self):
        # Backward pass
        dense_update_gradients(self)
        return None

    def update_parameters(self):
        # Update parameters
        dense_update_parameters(self)
        return None
