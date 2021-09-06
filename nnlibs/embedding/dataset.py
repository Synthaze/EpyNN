# EpyNN/nnlibs/embedding/dataset.py
# Standard library imports
import warnings

# Related third party imports
import numpy as np

# Local application/library specific imports
from nnlibs.commons.io import (
    encode_dataset,
    scale_features,
    index_elements_auto,
)
from nnlibs.commons.models import dataSet


def embedding_check(X_data, Y_data=None, X_scale=False):
    """Pre-processing.

    :param X_data: Set of sample features.
    :type encode: list[list] or :class:`numpy.ndarray`

    :param Y_data: Set of samples label.
    :type encode: list[list[int] or int] or :class:`numpy.ndarray`, optional

    :param X_scale: Set to True to normalize sample features within [0, 1].
    :type X_scale: bool, optional

    :return: Sample features and label.
    :rtype: tuple[:class:`numpy.ndarray`]
    """
    if X_scale:
        # Array-wide normalization in [0, 1]
        X_data = scale_features(X_data)

    X_data = np.array(X_data)

    Y_data = np.array(Y_data)

    return X_data, Y_data


def embedding_encode(layer, X_data, Y_data, X_encode, Y_encode):
    """One-hot encoding for samples features and label.

    :param layer: An instance of the :class:`nnlibs.embedding.models.Embedding`
    :type layer: :class:`nnlibs.embedding.models.Embedding`

    :param X_data: Set of sample features.
    :type encode: list[list] or :class:`numpy.ndarray`

    :param Y_data: Set of samples label.
    :type encode: list[list[int] or int] or :class:`numpy.ndarray`, optional

    :param X_encode: Set to True to one-hot encode features.
    :type encode: bool

    :param Y_encode: Set to True to one-hot encode labels.
    :type encode: bool

    :return:
    :rtype :
    """
    # Features one-hot encoding
    if X_encode:
        layer.e2i, layer.i2e, layer.d['e'] = index_elements_auto(X_data)
        X_data = encode_dataset(X_data, layer.e2i, layer.d['e'])
    # Label one-hot encoding
    if Y_encode:
        num_classes = len(list(set(Y_data.flatten())))
        Y_data = np.eye(num_classes)[Y_data]

    return X_data, Y_data


def embedding_prepare(layer, X_data, Y_data):
    """Prepare dataset for Embedding layer object.

    :param layer: An instance of the :class:`nnlibs.embedding.models.Embedding`
    :type layer: :class:`nnlibs.embedding.models.Embedding`

    :param X_data: Set of sample features.
    :type encode: list[list] or :class:`numpy.ndarray`

    :param Y_data: Set of samples label.
    :type encode: list[list[int] or int] or :class:`numpy.ndarray`, optional

    :return: All training, testing and validations sets along with batched training set
    :rtype: tuple[:class:`nnlibs.commons.models.dataSet`]
    """
    # Embedding parameters
    se_dataset = layer.se_dataset

    # Pair-wise features-label list
    dataset = list(zip(X_data, Y_data))

    # Split and separate features and label
    dtrain, dtest, dval = split_dataset(dataset, se_dataset)

    X_train, Y_train = zip(*dtrain)
    X_test, Y_test = zip(*dtest) if dtest else [(), ()]
    X_val, Y_val = zip(*dval) if dval else [(), ()]

    # Instantiate dataSet objects
    dtrain = dataSet(X_data=X_train, Y_data=Y_train, name='dtrain')
    dtest = dataSet(X_data=X_test, Y_data=Y_test, name='dtest')
    dval = dataSet(X_data=X_val, Y_data=Y_val, name='dval')

    embedded_data = (dtrain, dtest, dval)

    return embedded_data


def split_dataset(dataset, se_dataset):
    """Split dataset in training, testing and validation sets.

    :param dataset: Dataset containing samples features and label
    :type dataset: tuple[list or :class:`numpy.ndarray`]

    :param se_dataset: Settings for sets preparation
    :type se_dataset: dict[str: int]

    :return: Training, testing and validation sets.
    :rtype: tuple[list]
    """
    # Retrieve relative sizes
    dtrain_relative = se_dataset['dtrain_relative']
    dtest_relative = se_dataset['dtest_relative']
    dval_relative = se_dataset['dval_relative']

    # Compute absolute sizes with respect to full dataset
    sum_relative = sum([dtrain_relative, dtest_relative, dval_relative])

    dtrain_length = round(dtrain_relative / sum_relative * len(dataset))
    dtest_length = round(dtest_relative / sum_relative * len(dataset))
    dval_length = round(dval_relative / sum_relative * len(dataset))

    # Slice full dataset
    dtrain = dataset[:dtrain_length]
    dtest = dataset[dtrain_length:dtrain_length + dtest_length]
    dval = dataset[dtrain_length + dtest_length:]

    return dtrain, dtest, dval


def mini_batches(layer):
    """Shuffle and divide dataset in batches for each training epoch.

    :param layer: An instance of the :class:`nnlibs.embedding.models.Embedding`
    :type layer: :class:`nnlibs.embedding.models.Embedding`

    :return: Batches made from dataset with respect to batch_size
    :rtype: list[Object]
    """
    # Retrieve training set and make pair-wise features-label dataset
    dtrain = layer.dtrain
    dtrain = list(zip(dtrain.X.tolist(), dtrain.Y.tolist()))

    batch_size = layer.se_dataset['batch_size']

    # Shuffle dataset
    if hasattr(layer, 'np_rng'):
        warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
        layer.np_rng.shuffle(dtrain)
    else:
        np.random.shuffle(dtrain)

    # Compute number of batches w.r.t. batch_size
    if not batch_size:
        batch_size = len(dtrain)

    n_batch = len(dtrain) // batch_size

    if not n_batch:
        n_batch = 1

    batch_dtrain = []

    # Iterate over number of batches
    for i in range(n_batch):

        # Slice training set
        batch = dtrain[i * batch_size:(i+1) * batch_size]

        # Separate features and label
        X_batch, Y_batch = zip(*batch)

        # Append to list of training batches
        batch = dataSet(X_data=X_batch, Y_data=Y_batch, name=str(i))
        batch_dtrain.append(batch)

    return batch_dtrain
