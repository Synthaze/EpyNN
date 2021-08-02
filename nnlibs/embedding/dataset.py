# EpyNN/nnlibs/embedding/dataset.py
# Related third party imports
import numpy as np

# Local application/library specific imports
from nnlibs.commons.io import encode_dataset
from nnlibs.commons.models import dataSet


def embedding_prepare(layer, dataset, se_dataset, encode, single):
    """Prepare dataset for Embedding layer object

    :param layer: An instance of the :class:`nnlibs.embedding.models.Embedding`
    :type layer: class:`nnlibs.embedding.models.Embedding`

    :param dataset:
    :type dataset: list

    :param se_dataset:
    :type se_dataset: dict

    :param encode:
    :type encode: bool

    :param single:
    :type single: bool

    :return:
    :rtype : tuple
    """
    if encode == True:
        index_vocabulary_auto(layer, dataset)
        dataset = encoded_dataset = encode_dataset(dataset, layer.w2i, layer.d['v'])

    dtrain, dtest, dval = split_dataset(dataset, se_dataset)

    if single:
        dtrain = dataset

    batch_dtrain = mini_batches(dtrain, se_dataset)

    if se_dataset['dataset_name'] != None:
        suffix = '_' + se_dataset['dataset_name']
    else:
        pass

    dtrain = dataSet(dtrain, name='dtrain'+suffix)
    dtest = dataSet(dtest, name='dtest'+suffix)
    dval = dataSet(dval, name='dval'+suffix)

    for i, batch in enumerate(batch_dtrain):
        batch = dataSet(batch, name='dtrain_'+str(i)+suffix)
        batch_dtrain[i] = batch

    embedded_data = (dtrain, dtest, dval, batch_dtrain)

    return embedded_data


def index_vocabulary_auto(layer, dataset):
    """[Summary]

    :param layer: An instance of the :class:`nnlibs.embedding.models.Embedding`
    :type layer: class:`nnlibs.embedding.models.Embedding`

    :raises [ErrorType]: [ErrorDescription]

    :return: [ReturnDescription]
    :rtype: [ReturnType]
    """

    words = sorted(list(set([w for x in dataset for w in x[0]])))

    word_to_idx = layer.w2i = { k:i for i,k in enumerate(words) }
    idx_to_word = layer.i2w = { v:k for k,v in layer.w2i.items() }

    vocab_size = layer.d['v'] = len(layer.w2i.keys())

    return None


def split_dataset(dataset, se_dataset):
    """[Summary]

    :param dataset: [ParamDescription], defaults to [DefaultParamVal]
    :type dataset: list

    :param se_dataset: [ParamDescription], defaults to [DefaultParamVal]
    :type se_dataset: dict

    :return: [ReturnDescription]
    :rtype: [ReturnType]
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


def mini_batches(dataset, se_dataset):
    """[Summary]

    :param dataset: [ParamDescription], defaults to [DefaultParamVal]
    :type dataset: list

    :param se_dataset: [ParamDescription], defaults to [DefaultParamVal]
    :type se_dataset: dict

    :return: [ReturnDescription]
    :rtype: list
    """

    batch_size = se_dataset['batch_size']

    n_batch = len(dataset) // batch_size

    if not n_batch:
        n_batch = 1

    batch_dataset = []

    for i in range(n_batch):

        batch = dataset[i * batch_size:(i+1) * batch_size]

        batch_dataset.append(batch)

    return batch_dataset
