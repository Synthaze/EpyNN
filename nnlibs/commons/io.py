# EpyNN/nnlibs/commons/io.py
# Related third party imports
import numpy as np


def one_hot_encode(i, vocab_size):
    """

    :param i:
    :type i:

    :param vocab_size:
    :type vocab_size:

    :return:
    :rtype:
    """
    one_hot = np.zeros(vocab_size)

    one_hot[i] = 1.0

    return one_hot


def one_hot_encode_sequence(sequence, word_to_idx, vocab_size):
    """.

    :param sequence:
    :type sequence:

    :param word_to_idx:
    :type word_to_idx:

    :param vocab_size:
    :type vocab_size:

    :return:
    :rtype:
    """
    encoding = np.array([one_hot_encode(word_to_idx[word], vocab_size) for word in sequence])

    encoding = encoding.T

    return encoding


def one_hot_decode_sequence(sequence, idx_to_word):
    """.

    :param sequence:
    :type sequence:

    :param idx_to_word:
    :type idx_to_word:

    :return:
    :rtype:
    """
    decoding = [idx_to_word[np.argmax(encoded)] for encoded in sequence]

    return decoding


def encode_dataset(dataset, word_to_idx, vocab_size):
    """.

    :param dataset:
    :type dataset:

    :param word_to_idx:
    :type word_to_idx:

    :param vocab_size:
    :type vocab_size:

    :return:
    :rtype:
    """
    encoded_dataset = []

    for i in range(len(dataset)):

        features = dataset[i][0]

        label = dataset[i][1].copy()

        encoded_features = one_hot_encode_sequence(features, word_to_idx, vocab_size)

        sample = [encoded_features,label]

        encoded_dataset.append(sample)

    return encoded_dataset
