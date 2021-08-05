# EpyNN/nnlibs/embedding/dataset.py
# Related third party imports
import numpy as np

# Local application/library specific imports
from nnlibs.commons.io import encode_dataset
from nnlibs.commons.models import dataSet


def embedding_check(X_data, Y_data, X_scale=False):
    """Check validity of user input and apply pre-processing.

    :param X_data: Dataset containing samples features
    :type encode: list[list[list]]

    :param Y_data: Dataset containing samples label
    :type encode: list[list[list[int]]]

    :param X_scale: Set to True to normalize sample features within [0, 1]
    :type X_scale: bool
    """
    if type(X_data) == type(Y_data) == None:
        return None

    X_data = np.array(X_data)
    Y_data = np.array(Y_data)

    if X_scale:
        X_data = (X_data-np.min(X_data)) / (np.max(X_data)-np.min(X_data))

    return X_data, Y_data


def embedding_encode(layer, X_data, Y_data, X_encode, Y_encode):
    """One-hot encoding for samples features and label.

    :param layer: An instance of the :class:`nnlibs.embedding.models.Embedding`
    :type layer: :class:`nnlibs.embedding.models.Embedding`

    :param X_data: Dataset containing samples features
    :type encode: list[list[list]]

    :param Y_data: Dataset containing samples label
    :type encode: list[list[list[int]]]

    :param X_encode: Set to True to one-hot encode features
    :type encode: bool

    :param Y_encode: Set to True to one-hot encode labels
    :type encode: bool

    :return:
    :rtype :
    """
    # Features one-hot encoding
    if X_encode:
        layer.w2i, layer.i2w, layer.d['v'] = index_vocabulary_auto(X_data)
        X_data = X_encoded_dataset = encode_dataset(X_data, layer.w2i, layer.d['v'])
    # Label one-hot encoding
    if Y_encode:
        num_classes = len(list(set(Y_data.flatten())))
        Y_data = np.eye(num_classes)[Y_data]

    return X_data, Y_data


def embedding_prepare(layer, X_data, Y_data):
    """Prepare dataset for Embedding layer object.

    :param layer: An instance of the :class:`nnlibs.embedding.models.Embedding`
    :type layer: :class:`nnlibs.embedding.models.Embedding`

    :param X_data: Dataset containing samples features
    :type encode: list[list[list]]

    :param Y_data: Dataset containing samples label
    :type encode: list[list[list[int]]]

    :return: All training, testing and validations sets along with batched training set
    :rtype : tuple[:class:`nnlibs.commons.models.dataSet`]
    """
    se_dataset = layer.se_dataset

    dataset = list(zip(X_data, Y_data))

    dtrain, dtest, dval = split_dataset(dataset, se_dataset)

    batch_dtrain = mini_batches(dtrain, se_dataset['batch_size'])

    if se_dataset['dataset_name'] != None:
        suffix = '_' + se_dataset['dataset_name']
    else:
        pass

    X_train, Y_train = zip(*dtrain)
    X_test, Y_test = zip(*dtest) if dtest else [(), ()]
    X_val, Y_val = zip(*dval) if dval else [(), ()]

    dtrain = dataSet(X_data=X_train, Y_data=Y_train, name='dtrain' + suffix)
    dtest = dataSet(X_data=X_test, Y_data=Y_test, name='dtest' + suffix)
    dval = dataSet(X_data=X_val, Y_data=Y_val, name='dval' + suffix)

    for i, batch in enumerate(batch_dtrain):
        X_batch, Y_batch = zip(*batch)
        batch = dataSet(X_data=X_batch, Y_data=Y_batch, name='dtrain_' + str(i) + suffix)
        batch_dtrain[i] = batch

    embedded_data = (dtrain, dtest, dval, batch_dtrain)

    return embedded_data


def index_vocabulary_auto(X_or_Y_data):
    """Determine vocabulary size and generate dictionnary for one-hot encoding or features or label

    :param X_or_Y_data: Dataset containing samples features or samples label
    :type encode: list[list[list]]

    :return: One-hot encoding converter
    :rtype: dict

    :return: One-hot decoding converter
    :rtype: dict

    :return: Vocabulary size
    :rtype: int
    """
    words = sorted(list(set(X_or_Y_data.flatten())))

    word_to_idx = {w:i for i,w in enumerate(words)}
    idx_to_word = {i:w for w,i in word_to_idx.items()}

    vocab_size = len(word_to_idx.keys())

    return word_to_idx, idx_to_word, vocab_size


def split_dataset(dataset, se_dataset):
    """Split dataset in training, testing and validation sets.

    :param dataset: Dataset containing samples features and label
    :type dataset: list[list[list,list[int]]]

    :param se_dataset: Settings for sets preparation
    :type se_dataset: dict

    :return:
    :rtype:
    """

    dtrain_relative = se_dataset['dtrain_relative']
    dtest_relative = se_dataset['dtest_relative']
    dval_relative = se_dataset['dval_relative']

    sum_relative = sum([dtrain_relative, dtest_relative, dval_relative])

    dtrain_length = round(dtrain_relative / sum_relative * len(dataset))
    dtest_length = round(dtest_relative / sum_relative * len(dataset))
    dval_length = round(dval_relative / sum_relative * len(dataset))

    dtrain = dataset[:dtrain_length]
    dtest = dataset[dtrain_length:dtrain_length + dtest_length]
    dval = dataset[dtrain_length + dtest_length:]

    return dtrain, dtest, dval


def mini_batches(dataset, batch_size):
    """Divide dataset in batches.

    :param dataset: Dataset containing samples features and label
    :type dataset: list[list[list,list[int]]]

    :param batch_size: Number of samples per batch
    :type se_dataset: int

    :return: Batches made from dataset with respect to batch_size
    :rtype: list[list[list[list,list[int]]]]
    """
    if not batch_size:
        batch_size = len(dataset)

    n_batch = len(dataset) // batch_size

    if not n_batch:
        n_batch = 1

    batch_dataset = []

    for i in range(n_batch):

        batch = dataset[i * batch_size:(i+1) * batch_size]

        batch_dataset.append(batch)

    return batch_dataset
