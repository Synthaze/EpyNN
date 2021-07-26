# EpyNN/nnlibs/embedding/parameters.py
# Local application/library specific imports
from nnlibs.commons.models import Layer
from nnlibs.embedding.dataset import embedding_prepare
from nnlibs.embedding.forward import embedding_forward
from nnlibs.embedding.backward import embedding_backward
from nnlibs.embedding.parameters import (
    embedding_compute_shapes,
    embedding_initialize_parameters,
    embedding_update_gradients,
    embedding_update_parameters
)
import nnlibs.settings as se


class Embedding(Layer):
    """
    Definition of an Embedding Layer prototype

    Attributes
    ----------
    dtrain : numpy.ndarray
        .
    batch_dtrain : list
        .
    dtest : numpy.ndarray
        .
    dval : numpy.ndarray
        .
    dsets : list
        .

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
                dataset,
                se_dataset=se.dataset,
                encode=False):

        super().__init__()

        embedded_data = embedding_prepare(self, dataset, se_dataset, encode)

        self.dtrain, self.dtest, self.dval, self.batch_dtrain = embedded_data

        self.dsets = [self.dtrain, self.dtest, self.dval]

    def compute_shapes(self, A):
        embedding_compute_shapes(self, A)
        return None

    def initialize_parameters(self):
        embedding_initialize_parameters(self)
        return None

    def forward(self, A):
        # Forward pass
        A = embedding_forward(self, A)
        self.update_shapes(mode='forward')
        return A

    def backward(self, dA):
        # Backward pass
        dA = embedding_backward(self, dA)
        self.update_shapes(mode='backward')
        return dA

    def update_gradients(self):
        embedding_update_gradients(self)
        return None

    def update_parameters(self):
        embedding_update_parameters(self)
        return None
