# EpyNN/nnlibs/embedding/parameters.py
# Related third party imports
import numpy as np

# Local application/library specific imports
from nnlibs.commons.models import Layer
from nnlibs.embedding.dataset import embedding_prepare, embedding_check, embedding_encode
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

    :param dataset: Dataset containing samples features and label
    :type dataset: list[list[list,list[int]]]

    :param X_dataset: Dataset containing samples features
    :type encode: list[list[list]]

    :param Y_dataset: Dataset containing samples label
    :type encode: list[list[list[int]]]

    :param se_dataset: Settings for sets preparation
    :type se_dataset: dict

    :param X_encode: Set to True to one-hot encode features
    :type encode: bool

    :param Y_encode: Set to True to one-hot encode labels
    :type encode: bool

    :param single: Set to True to run only with training set
    :type single: bool

    :param X_scale: Set to True to normalize sample features within [0, 1]
    :type X_scale: bool
    """

    def __init__(self,
                X_dataset=None,
                Y_dataset=None,
                relative_size=(2, 1, 1),
                batch_size=None,
                X_encode=False,
                Y_encode=False,
                X_scale=False,
                name='dummy'):

        super().__init__()

        self.se_dataset = {
            'dtrain_relative': relative_size[0],
            'dtest_relative': relative_size[1],
            'dval_relative': relative_size[2],
            'batch_size': batch_size,
            'X_scale': X_scale,
            'X_encode': X_encode,
            'Y_encode': Y_encode,
            'dataset_name': name,
        }

        X_dataset, Y_dataset = embedding_check(X_dataset, Y_dataset, X_scale)

        X_dataset, Y_dataset = embedding_encode(self, X_dataset, Y_dataset, X_encode, Y_encode)

        embedded_data = embedding_prepare(self, X_dataset, Y_dataset)

        self.dtrain, self.dtest, self.dval, self.batch_dtrain = embedded_data

        self.dsets = [self.dtrain, self.dtest, self.dval]

        self.dsets = [dset for dset in self.dsets if dset.active]

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
