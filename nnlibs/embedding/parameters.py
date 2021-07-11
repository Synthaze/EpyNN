#EpyNN/nnlibs/embedding/parameters.py
import nnlibs.commons.io as cio

import numpy as np


def split_dataset(dataset,settings_datasets):

    dtrain_relative = settings_datasets['dtrain_relative']
    dtest_relative = settings_datasets['dtest_relative']
    dval_relative = settings_datasets['dval_relative']

    sum_relative = sum([dtrain_relative,dtest_relative,dval_relative])

    dtrain_length = round(dtrain_relative / sum_relative * len(dataset))
    dtest_length = round(dtest_relative / sum_relative * len(dataset))
    dval_length = round(dval_relative / sum_relative * len(dataset))

    dtrain = dataset[:dtrain_length]
    dtest = dataset[dtrain_length:dtrain_length+dtest_length]
    dval = dataset[dtrain_length+dtest_length:]

    return dtrain, dtest, dval


def encode_dataset(layer,dataset):

    words = set([ w for x in dataset for w in x[0] ])

    word_to_idx = layer.w2i = { k:i for i,k in enumerate(list(words)) }
    idx_to_word = layer.i2w = { v:k for k,v in layer.w2i.items() }

    vocab_size = layer.d['v'] = len(layer.w2i.keys())

    encoded_dataset = []

    for i in range(len(dataset)):

        features = dataset[i][0]
        label = dataset[i][1].copy()

        encoded_features = cio.one_hot_encode_sequence(features,word_to_idx,vocab_size)

        sample = [encoded_features,label]

        encoded_dataset.append(sample)

    return encoded_dataset


def mini_batches(dataset,settings_datasets):

    n_batch = settings_datasets['batch_number']

    batch_dataset = []

    for i in range(n_batch):

        start = len(dataset) * i // n_batch
        stop = len(dataset) * (i + 1) // n_batch

        batch = dataset[start:stop]

        batch_dataset.append(batch)

    return batch_dataset


def object_vectorize(dataset,type=str(),prefix=str()):

    # Prepare X data
    x_data = [ x[0] for x in dataset ]
    # Prepare Y data
    y_data = [ x[1] for x in dataset ]

    if prefix != str():
        name = prefix + '_' + type
    else:
        name = type

    dset = lambda: None

    # Identifier
    dset.n = name

    # Set X and Y data for training in corresponding uppercase attributes
    dset.X = np.array(x_data)
    dset.Y = np.array(y_data)

    # Restore Y data as single digit labels
    dset.y = np.argmax(dset.Y,axis=1)

    # Set numerical id for each sample
    dset.id = np.array([ i for i in range(len(dataset)) ])

    # Number of samples
    dset.s = str(len(dset.id))

    ## Predicted data
    # Output of forward propagation
    dset.A = None
    # Predicted labels
    dset.P = None

    return dset


def init_shapes(layer):

    # Set layer shapes

    return None


def init_forward(layer,A):

    # Set and cache layer X and X.shape
    X = layer.fc['X'] = A
    layer.fs['X'] = X.shape

    return X


def init_backward(layer,dA):

    # Cache dX (current) from dA (prev)
    dX = layer.bc['dX'] = dA

    return dX


def init_params(layer):

    # Init parameters with corresponding function

    layer.init = False

    return None
