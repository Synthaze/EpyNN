# EpyNN/epynn/embedding/parameters.py
# Related third party imports
import numpy as np

# Local application/library specific imports
from epynn.commons.models import Layer
from epynn.embedding.dataset import (
    embedding_prepare,
    embedding_encode,
    embedding_check,
    mini_batches,
)
from epynn.embedding.forward import embedding_forward
from epynn.embedding.backward import embedding_backward
from epynn.embedding.parameters import (
    embedding_compute_shapes,
    embedding_initialize_parameters,
    embedding_compute_gradients,
    embedding_update_parameters
)


class Embedding(Layer):
    """
    Definition of an embedding layer prototype.

    :param X_data: Dataset containing samples features, defaults to `None` which returns an empty layer.
    :type X_data: list[list[float or str or list[float or str]]] or NoneType, optional

    :param Y_data: Dataset containing samples label, defaults to `None`.
    :type Y_data: list[int or list[int]] or NoneType, optional

    :param relative_size: For training, validation and testing sets. Defaults to `(2, 1, 1)`
    :type relative_size: tuple[int], optional

    :param batch_size: For training batches, defaults to None which makes a single batch out of the training data.
    :type batch_size: int or NoneType, optional

    :param X_encode: Set to True to one-hot encode features, default to `False`.
    :type encode: bool, optional

    :param Y_encode: Set to True to one-hot encode labels, default to `False`.
    :type encode: bool, optional

    :param X_scale: Normalize sample features within [0, 1], default to `False`.
    :type X_scale: bool, optional
    """

    def __init__(self,
                 X_data=None,
                 Y_data=None,
                 relative_size=(2, 1, 0),
                 batch_size=None,
                 X_encode=False,
                 Y_encode=False,
                 X_scale=False):
        """Initialize instance variable attributes.
        """
        super().__init__()

        self.se_dataset = {
            'dtrain_relative': relative_size[0],
            'dval_relative': relative_size[1],
            'dtest_relative': relative_size[2],
            'batch_size': batch_size,
            'X_scale': X_scale,
            'X_encode': X_encode,
            'Y_encode': Y_encode,
        }

        X_data, Y_data = embedding_check(X_data, Y_data, X_scale)

        X_data, Y_data = embedding_encode(self, X_data, Y_data, X_encode, Y_encode)

        embedded_data = embedding_prepare(self, X_data, Y_data)

        self.dtrain, self.dval, self.dtest = embedded_data

        # Keep non-empty datasets
        self.dsets = [self.dtrain, self.dval, self.dtest]
        self.dsets = [dset for dset in self.dsets if dset.active]

        self.trainable = False

        return None

    def training_batches(self, init=False):
        """Wrapper for :func:`epynn.embedding.dataset.mini_batches()`.

        :param init: Wether to prepare a zip of X and Y data, defaults to False.
        :type init: bool, optional
        """
        if init:
            self.dtrain_zip = list(zip(self.dtrain.X, self.dtrain.Y))

        self.batch_dtrain = mini_batches(self)

        return None

    def compute_shapes(self, A):
        """Wrapper for :func:`epynn.embedding.parameters.embedding_compute_shapes()`.

        :param A: Output of forward propagation from previous layer.
        :type A: :class:`numpy.ndarray`
        """
        embedding_compute_shapes(self, A)

        return None

    def initialize_parameters(self):
        """Wrapper for :func:`epynn.embedding.parameters.embedding_initialize_parameters()`.
        """
        embedding_initialize_parameters(self)

        return None

    def forward(self, A):
        """Wrapper for :func:`epynn.embedding.forward.embedding_forward()`.

        :param A: Output of forward propagation from *previous* layer.
        :type A: :class:`numpy.ndarray`

        :return: Output of forward propagation for **current** layer.
        :rtype: :class:`numpy.ndarray`
        """
        A = embedding_forward(self, A)
        self.update_shapes(self.fc, self.fs)

        return A

    def backward(self, dX):
        """Wrapper for :func:`epynn.embedding.backward.embedding_backward()`.

        :param dX: Output of backward propagation from next layer.
        :type dX: :class:`numpy.ndarray`

        :return: Output of backward propagation for current layer.
        :rtype: :class:`numpy.ndarray`
        """
        dX = embedding_backward(self, dX)
        self.update_shapes(self.bc, self.bs)

        return dX

    def compute_gradients(self):
        """Wrapper for :func:`epynn.embedding.parameters.embedding_compute_gradients()`. Dummy method, there are no gradients to compute in layer.
        """
        embedding_compute_gradients(self)

        return None

    def update_parameters(self):
        """Wrapper for :func:`epynn.embedding.parameters.embedding_update_parameters()`. Dummy method, there are no parameters to update in layer.
        """
        if self.trainable:
            embedding_update_parameters(self)

        return None
