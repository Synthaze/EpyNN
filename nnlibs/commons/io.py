# EpyNN/nnlibs/commons/io.py
# Related third party imports
import numpy as np


def one_hot_encode(i, vocab_size):
    """Generate one-hot encoding array.

    :param i: One-hot index for current word
    :type i: int

    :param vocab_size: Number of keys in the word to index encoder
    :type vocab_size: int

    :return: One-hot encoding array for current word
    :rtype: :class:`numpy.ndarray`
    """
    one_hot = np.zeros(vocab_size)

    one_hot[i] = 1.0

    return one_hot


def one_hot_encode_sequence(sequence, word_to_idx, vocab_size):
    """One-hot encode sequence.

    :param sequence: Sequential data
    :type sequence: list

    :param word_to_idx: Converter with word as key and index as value
    :type word_to_idx: dict

    :param vocab_size: Number of keys in converter
    :type vocab_size: int

    :return: One-hot encoded sequence
    :rtype: :class:`numpy.ndarray`
    """
    encoding = np.array([one_hot_encode(word_to_idx[word], vocab_size) for word in sequence])

    encoding = encoding

    return encoding


def one_hot_decode_sequence(sequence, idx_to_word):
    """One-hot decode sequence.

    :param sequence: One-hot encoded sequence
    :type sequence: :class:`numpy.ndarray`

    :param idx_to_word: Converter with index as key and word as value
    :type idx_to_word:

    :return: One-hot decoded sequence
    :rtype: :class:`numpy.ndarray`
    """
    decoding = [idx_to_word[np.argmax(encoded)] for encoded in sequence]

    return decoding


def encode_dataset(X_dataset, word_to_idx, vocab_size):
    """One-hot encode features from samples in dataset.

    :param dataset: Contains samples
    :type dataset: list[list[list,list]]

    :param word_to_idx: Converter with word as key and index as value
    :type word_to_idx: dict

    :param vocab_size: Number of keys in converter
    :type vocab_size: int

    :return: One-hot encoded dataset
    :rtype:
    """
    X_dataset = []

    for i in range(X_dataset.shape[0]):

        sequence = X_dataset[i]

        encoded_sequence = one_hot_encode_sequence(sequence, word_to_idx, vocab_size)

        X_dataset.append(encoded_sequence)

    X_dataset = np.array(X_dataset)

    return X_dataset
