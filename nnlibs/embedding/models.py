# EpyNN/nnlibs/embedding/parameters.py
# Local application/library specific imports
from nnlibs.commons.models import Layer
from nnlibs.embedding.dataset import embedding_prepare
from nnlibs.embedding.forward import embedding_forward
from nnlibs.embedding.backward import embedding_backward
from nnlibs.embedding.parameters import (
    embedding_compute_shapes,
    embedding_initialize_parameters,
    embedding_compute_gradients,
    embedding_update_parameters
)
import nnlibs.settings as se


class Embedding(Layer):
    """
    Definition of an embedding layer prototype.

    :param dataset: Samples as list of features and label.
    :type dataset: list

    :param se_dataset: Settings to embed data in layer.
    :type se_dataset: dict

    :param encode: One-hot encoding of features.
    :type encode: bool
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
        self.update_shapes(self.fc, self.fs)
        return A

    def backward(self, dA):
        # Backward pass
        dA = embedding_backward(self, dA)
        self.update_shapes(self.bc, self.bs)
        return dA

    def compute_gradients(self):
        embedding_compute_gradients(self)
        return None

    def update_parameters(self):
        embedding_update_parameters(self)
        return None
