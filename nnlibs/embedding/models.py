# EpyNN/nnlibs/embedding/parameters.py
# Related third party imports
import numpy as np

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


class Embedding(Layer):
    """
    Definition of an embedding layer prototype.

    :param dataset: Samples as list of features and label.
    :type dataset: list

    :param se_dataset: Settings to embed data in layer.
    :type se_dataset: dict

    :param encode: One-hot encoding of features.
    :type encode: bool

    :param single:
    :type single: bool
    """

    def __init__(self,
                dataset=None,
                se_dataset=None,
                batch_size=None,
                encode=False,
                single=False,
                scale=False):

        super().__init__()

        if not dataset:
            return None

        if batch_size:
            se_dataset['batch_size'] = batch_size

        if scale:
            x_data = np.array([x[0] for x in dataset])
            x_data = (x_data-np.min(x_data)) / (np.max(x_data)-np.min(x_data))
            y_data = [x[1] for x in dataset]
            dataset = [[x, y] for x,y in zip(x_data, y_data)]

        embedded_data = embedding_prepare(self, dataset, se_dataset, encode, single)

        self.dtrain, self.dtest, self.dval, self.batch_dtrain = embedded_data

        self.dsets = [self.dtrain, self.dtest, self.dval]

        if single:
            self.dsets = [self.dtrain]

        self.single = single

        return None

    def compute_shapes(self, A):
        """Wrapper for :func:`nnlibs.embedding.parameters.embedding_compute_shapes()`.
        """
        embedding_compute_shapes(self, A)

        return None

    def initialize_parameters(self):
        """Wrapper for :func:`nnlibs.embedding.parameters.embedding_initialize_parameters()`.
        """
        embedding_initialize_parameters(self)

        return None

    def forward(self, A):
        """Wrapper for :func:`nnlibs.embedding.forward.embedding_forward()`.
        """
        A = embedding_forward(self, A)
        self.update_shapes(self.fc, self.fs)

        return A

    def backward(self, dA):
        """Wrapper for :func:`nnlibs.embedding.backward.embedding_backward()`.
        """
        dA = embedding_backward(self, dA)
        self.update_shapes(self.bc, self.bs)

        return dA

    def compute_gradients(self):
        """Wrapper for :func:`nnlibs.embedding.parameters.embedding_compute_gradients()`.
        """
        embedding_compute_gradients(self)

        return None

    def update_parameters(self):
        """Wrapper for :func:`nnlibs.embedding.parameters.embedding_update_parameters()`.
        """
        embedding_update_parameters(self)

        return None
