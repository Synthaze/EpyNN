# EpyNN/nnlibs/pool/models.py
# Related third party imports
import numpy as np

# Local application/library specific imports
from nnlibs.commons.models import Layer
from nnlibs.pooling.forward import pooling_forward
from nnlibs.pooling.backward import pooling_backward
from nnlibs.pooling.parameters import (
    pooling_compute_shapes,
    pooling_initialize_parameters,
    pooling_update_gradients,
    pooling_update_parameters
)


class Pooling(Layer):
    """
    Definition of a Pooling Layer prototype

    Attributes
    ----------
    initialization : function
        Function used for weight initialization.
    activation : dict
        Activation functions as key-value pairs for log purpose.
    activate : function
        Activation function.
    lrate : list
        Learning rate along epochs for Pooling layer

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
                f_width,
                stride=1,
                pool=np.max):

        super().__init__()

        self.activation = { 'pool': np.max.__name__ }

        self.pool = pool

        self.d['w'] = f_width
        self.d['s'] = stride

        self.lrate = []

    def compute_shapes(self, A):
        pooling_compute_shapes(self, A)
        return None

    def initialize_parameters(self):
        pooling_initialize_parameters(self)
        return None

    def forward(self, A):
        # Forward pass
        self.compute_shapes(A)
        A = pooling_forward(self, A)
        self.update_shapes(mode='forward')
        return A

    def backward(self, dA):
        # Backward pass
        dA = pooling_backward(self, dA)
        self.update_shapes(mode='backward')
        return dA

    def update_gradients(self):
        # Backward pass
        pooling_update_gradients(self)
        return None

    def update_parameters(self):
        # Update parameters
        pooling_update_parameters(self)
        return None


    def assemble_block(self, block, t, b, l, r):

        im, ih, iw, id = self.fs['X']

        block = np.repeat(block, self.d['w'] ** 2, 2)

        block = np.array(np.split(block, block.shape[2] / self.d['w'], 2))
        block = np.moveaxis(block, 0, 2)

        block = np.array(np.split(block, block.shape[2] / self.d['w'], 2))
        block = np.moveaxis(block, 0, 3)

        return np.reshape(block, ( im, ih - t - b, iw - l - r,  id))
