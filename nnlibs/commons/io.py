#EpyNN/nnlibs/commons/io.py
import numpy as np


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


def dsets_padding(dsets,padding=0):

    for dset in dsets:

        dset.X = dset.X.T

        shape = ((0, 0), (padding, padding), (padding, padding))

        dset.X = np.pad(dset.X, shape, mode='constant', constant_values = (0, 0))

    return dsets
