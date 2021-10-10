# EpyNN/epynn/embedding/dataset.py
# Standard library imports
import warnings

# Related third party imports
import numpy as np

# Local application/library specific imports
from epynn.commons.io import (
    encode_dataset,
    scale_features,
    index_elements_auto,
)
from epynn.commons.models import dataSet


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

    :param layer: An instance of the :class:`epynn.embedding.models.Embedding`
    :type layer: :class:`epynn.embedding.models.Embedding`

    :param X_data: Set of sample features.
    :type encode: list[list] or :class:`numpy.ndarray`

    :param Y_data: Set of samples label.
    :type encode: list[list[int] or int] or :class:`numpy.ndarray`

    :param X_encode: Set to True to one-hot encode features.
    :type encode: bool

    :param Y_encode: Set to True to one-hot encode labels.
    :type encode: bool

    :return: Encoded set of sample features, if applicable.
    :rtype : :class:`numpy.ndarray`

    :return: Encoded set of sample label, if applicable.
    :rtype : :class:`numpy.ndarray`
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

    :param layer: An instance of the :class:`epynn.embedding.models.Embedding`
    :type layer: :class:`epynn.embedding.models.Embedding`

    :param X_data: Set of sample features.
    :type encode: list[list] or :class:`numpy.ndarray`

    :param Y_data: Set of samples label.
    :type encode: list[list[int] or int] or :class:`numpy.ndarray`

    :return: All training, validation and testing sets along with batched training set
    :rtype: tuple[:class:`epynn.commons.models.dataSet`]
    """
    # Embedding parameters
    se_dataset = layer.se_dataset

    # Pair-wise features-label list
    dataset = list(zip(X_data, Y_data))

    # Split and separate features and label
    dtrain, dval, dtest = split_dataset(dataset, se_dataset)

    X_train, Y_train = zip(*dtrain)
    X_val, Y_val = zip(*dval) if dval else [(), ()]
    X_test, Y_test = zip(*dtest) if dtest else [(), ()]

    # Instantiate dataSet objects
    dtrain = dataSet(X_data=X_train, Y_data=Y_train, name='dtrain')
    dval = dataSet(X_data=X_val, Y_data=Y_val, name='dval')
    dtest = dataSet(X_data=X_test, Y_data=Y_test, name='dtest')

    embedded_data = (dtrain, dval, dtest)

    return embedded_data


def split_dataset(dataset, se_dataset):
    """Split dataset in training, testing and validation sets.

    :param dataset: Dataset containing sample features and label
    :type dataset: tuple[list or :class:`numpy.ndarray`]

    :param se_dataset: Settings for sets preparation
    :type se_dataset: dict[str: int]

    :return: Training, testing and validation sets.
    :rtype: tuple[list]
    """
    # Retrieve relative sizes
    dtrain_relative = se_dataset['dtrain_relative']
    dval_relative = se_dataset['dval_relative']
    dtest_relative = se_dataset['dtest_relative']

    # Compute absolute sizes with respect to full dataset
    sum_relative = sum([dtrain_relative, dval_relative, dtest_relative])

    dtrain_length = round(dtrain_relative / sum_relative * len(dataset))
    dval_length = round(dval_relative / sum_relative * len(dataset))
    dtest_length = round(dtest_relative / sum_relative * len(dataset))

    # Slice full dataset
    dtrain = dataset[:dtrain_length]
    dval = dataset[dtrain_length:dtrain_length + dval_length]
    dtest = dataset[dtrain_length + dval_length:]

    return dtrain, dval, dtest


def mini_batches(layer):
    """Shuffle and divide dataset in batches for each training epoch.

    :param layer: An instance of the :class:`epynn.embedding.models.Embedding`
    :type layer: :class:`epynn.embedding.models.Embedding`

    :return: Batches made from dataset with respect to batch_size
    :rtype: list[Object]
    """
    # Retrieve training set and make pair-wise features-label dataset
    dtrain_zip = layer.dtrain_zip

    batch_size = layer.se_dataset['batch_size']

    # Shuffle dataset
    if hasattr(layer, 'np_rng'):
        warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
        layer.np_rng.shuffle(dtrain_zip)
    else:
        np.random.shuffle(dtrain_zip)

    # Compute number of batches w.r.t. batch_size
    if not batch_size:
        batch_size = len(dtrain_zip)

    n_batch = len(dtrain_zip) // batch_size

    if not n_batch:
        n_batch = 1

    # Slice to make sure split will result in equal division
    dtrain_zip = dtrain_zip[: n_batch * batch_size]

    X_train, Y_train = zip(*dtrain_zip)

    X_train = np.split(np.array(X_train), n_batch, axis=0)
    Y_train = np.split(np.array(Y_train), n_batch, axis=0)

    # Set into dataSet object
    batch_dtrain = [dataSet(X_data=X_batch, Y_data=Y_batch, name=str(i))
                    for i, (X_batch, Y_batch) in enumerate(zip(X_train, Y_train))]

    return batch_dtrain
