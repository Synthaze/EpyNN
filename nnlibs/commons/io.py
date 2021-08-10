# EpyNN/nnlibs/commons/io.py
# Related third party imports
import numpy as np


def index_vocabulary_auto(X_data):
    """Determine vocabulary size and generate dictionnary for one-hot encoding or features or label

    :param X_data: Dataset containing samples features or samples label
    :type X_data: list[list[list]]

    :return: One-hot encoding converter
    :rtype: dict

    :return: One-hot decoding converter
    :rtype: dict

    :return: Vocabulary size
    :rtype: int
    """
    X_data = X_data.flatten().tolist()

    words = sorted(list(set(X_data)))

    word_to_idx = {w:i for i,w in enumerate(words)}
    idx_to_word = {i:w for w,i in word_to_idx.items()}

    vocab_size = len(word_to_idx.keys())

    return word_to_idx, idx_to_word, vocab_size


def scale_features(X_data):
    """.
    """
    X_data = (X_data-np.min(X_data)) / (np.max(X_data)-np.min(X_data))

    return X_data


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


def encode_dataset(X_data, word_to_idx, vocab_size):
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
    X_encoded = []

    for i in range(X_data.shape[0]):

        sequence = X_data[i]

        encoded_sequence = one_hot_encode_sequence(sequence, word_to_idx, vocab_size)

        X_encoded.append(encoded_sequence)

    return X_encoded
