#EpyNN/nnlibs/commons/io.py
from nnlibs.commons.models import dataSet

import numpy as np
import copy

def one_hot_encode(i,vocab_size):

    one_hot = np.zeros(vocab_size)

    one_hot[i] = 1.0

    return one_hot


def one_hot_encode_sequence(sequence,word_to_idx,vocab_size):

    encoding = np.array([one_hot_encode(word_to_idx[word], vocab_size) for word in sequence])

    return encoding


def one_hot_decode_sequence(sequence,runData):

    decoding = [ runData.e['i2w'][np.argmax(encoded)] for encoded in sequence]

    decoding = ''.join(decoding)

    return decoding


def one_hot_encode_dataset(dataset,runData):

    word_to_idx(dataset,runData)

    for i in range(len(dataset)):

        dataset[i][0] = one_hot_encode_sequence(dataset[i][0],runData.e['w2i'],runData.e['v'])

    return dataset


def dsets_padding(dsets,padding=0):

    for dset in dsets:

        dset.X = dset.X.T

        shape = ((0, 0), (padding, padding), (padding, padding))

        dset.X = np.pad(dset.X, shape, mode='constant', constant_values = (0, 0))

    return dsets


def mini_batches(dset,hPars):

    batch_dset = []

    for i in range(hPars.b):

        start = dset.X.shape[0] * i // hPars.b
        stop = dset.X.shape[0] * (i + 1) // hPars.b

        X = dset.X[start:stop]
        Y = dset.Y[start:stop]

        batch = [ [x,y] for x,y in zip(X,Y) ]

        batch = dataSet(batch,dset.n+'_'+str(i))

        batch_dset.append(copy.deepcopy(batch))

    return batch_dset


def dataset_to_dsets(dataset,runData,prefix=None,encode=False):

    if encode == True:
        dataset = one_hot_encode_dataset(dataset,runData)

    dtrain = dataset[:len(dataset)//2]
    dtest = dataset[len(dataset)//2:len(dataset)//2+len(dataset)//4]
    dval = dataset[len(dataset)//2+len(dataset)//4:]

    if prefix != None:
        prefix = prefix + '_'
    else:
        prefix = ''

    # Initialization of dataSet() objects defined in nnlibs.commons.models
    dtrain = dataSet(dtrain,prefix+'dtrain')
    dtest = dataSet(dtest,prefix+'dtest')
    dval = dataSet(dval,prefix+'dval')

    dsets = [dtrain,dtest,dval]

    return dsets


def word_to_idx(dataset,runData):

    words = set(''.join([ x[0] for x in dataset ]))

    runData.e['w2i'] = { k:i for i,k in enumerate(list(words)) }
    runData.e['i2w'] = { v:k for k,v in runData.e['w2i'].items() }

    runData.e['v'] = len(runData.e['w2i'].keys())

    return None
