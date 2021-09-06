# EpyNN/nnlibs/embedding/parameters.py
# Related third party imports
import numpy as np

# Local application/library specific imports
from nnlibs.commons.models import Layer
from nnlibs.embedding.dataset import (
    embedding_prepare,
    embedding_encode,
    embedding_check,
    mini_batches,
)
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

    :param X_data: Dataset containing samples features.
    :type X_data: list[list[float or str or list[float or str]]]

    :param Y_data: Dataset containing samples label.
    :type Y_data: list[int or list[int]]

    :param relative_size: For training, testing and validation sets.
    :type relative_size: tuple[int]

    :param X_encode: Set to True to one-hot encode features.
    :type encode: bool

    :param Y_encode: Set to True to one-hot encode labels.
    :type encode: bool

    :param X_scale: Normalize sample features within [0, 1].
    :type X_scale: bool
    """

    def __init__(self,
                 X_data=None,
                 Y_data=None,
                 relative_size=(2, 1, 1),
                 batch_size=None,
                 X_encode=False,
                 Y_encode=False,
                 X_scale=False):
        """Initialize instance variable attributes.
        """
        super().__init__()

        self.se_dataset = {
            'dtrain_relative': relative_size[0],
            'dtest_relative': relative_size[1],
            'dval_relative': relative_size[2],
            'batch_size': batch_size,
            'X_scale': X_scale,
            'X_encode': X_encode,
            'Y_encode': Y_encode,
        }

        X_data, Y_data = embedding_check(X_data, Y_data, X_scale)

        X_data, Y_data = embedding_encode(self, X_data, Y_data, X_encode, Y_encode)

        embedded_data = embedding_prepare(self, X_data, Y_data)

        self.dtrain, self.dtest, self.dval = embedded_data

        # Keep non-empty datasets
        self.dsets = [self.dtrain, self.dtest, self.dval]
        self.dsets = [dset for dset in self.dsets if dset.active]

        self.trainable = False

        return None

    def training_batches(self):
        """Wrapper for :func:`nnlibs.embedding.dataset.mini_batches()`.
        """
        self.batch_dtrain = mini_batches(self)

        return None

    def compute_shapes(self, A):
        """Wrapper for :func:`nnlibs.embedding.parameters.embedding_compute_shapes()`.

        :param A: Output of forward propagation from previous layer.
        :type A: :class:`numpy.ndarray`
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

        :param A: Output of forward propagation from *previous* layer.
        :type A: :class:`numpy.ndarray`

        :return: Output of forward propagation for **current** layer.
        :rtype: :class:`numpy.ndarray`
        """
        A = embedding_forward(self, A)
        self.update_shapes(self.fc, self.fs)

        return A

    def backward(self, dX):
        """Wrapper for :func:`nnlibs.embedding.backward.embedding_backward()`.

        :param dX: Output of backward propagation from next layer.
        :type dX: :class:`numpy.ndarray`

        :return: Output of backward propagation for current layer.
        :rtype: :class:`numpy.ndarray`
        """
        dX = embedding_backward(self, dX)
        self.update_shapes(self.bc, self.bs)

        return dX

    def compute_gradients(self):
        """Wrapper for :func:`nnlibs.embedding.parameters.embedding_compute_gradients()`. Dummy method, there is no gradients to compute in layer.
        """
        embedding_compute_gradients(self)

        return None

    def update_parameters(self):
        """Wrapper for :func:`nnlibs.embedding.parameters.embedding_update_parameters()`. Dummy method, there is no parameters to update in layer.
        """
        if self.trainable:
            embedding_update_parameters(self)

        return None
