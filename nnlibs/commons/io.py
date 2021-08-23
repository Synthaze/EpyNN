# EpyNN/nnlibs/commons/io.py
# Related third party imports
import numpy as np


def index_vocabulary_auto(X_data):
    """Determine vocabulary size and generate dictionnary for one-hot encoding or features or label.

    :param X_data: Dataset containing samples features or samples label.
    :type X_data: :class:`numpy.ndarray`

    :return: One-hot encoding converter.
    :rtype: dict[str or int or float, int]

    :return: One-hot decoding converter.
    :rtype: dict[int, str or int or float]

    :return: Vocabulary size.
    :rtype: int
    """
    X_data = X_data.flatten().tolist()    # All vocabulary elements in 1D list

    words = sorted(list(set(X_data)))     # Unique vocabulary list
    vocab_size = len(words)               # Number of words

    # Converters to encode and decode sequences
    word_to_idx = {w: i for i, w in enumerate(words)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}

    return word_to_idx, idx_to_word, vocab_size


def scale_features(X_data):
    """Scale input array within [0, 1].

    :param X_data: Raw data.
    :type X_data: :class:`numpy.ndarray`

    :return: Normalized data.
    :rtype: :class:`numpy.ndarray`
    """
    X_data = (X_data-np.min(X_data)) / (np.max(X_data)-np.min(X_data))

    return X_data


def one_hot_encode(i, vocab_size):
    """Generate one-hot encoding array.

    :param i: One-hot index for current word.
    :type i: int

    :param vocab_size: Number of keys in the word to index encoder.
    :type vocab_size: int

    :return: One-hot encoding array for current word.
    :rtype: :class:`numpy.ndarray`
    """
    one_hot = np.zeros(vocab_size)

    one_hot[i] = 1.0    # Set 1 at index assigned to word

    return one_hot


def one_hot_encode_sequence(sequence, word_to_idx, vocab_size):
    """One-hot encode sequence.

    :param sequence: Sequential data.
    :type sequence: list or :class:`numpy.ndarray`

    :param word_to_idx: Converter with word as key and index as value.
    :type word_to_idx: dict[str or int or float, int]

    :param vocab_size: Number of keys in converter.
    :type vocab_size: int

    :return: One-hot encoded sequence.
    :rtype: :class:`numpy.ndarray`
    """
    encoding = np.array([one_hot_encode(word_to_idx[word], vocab_size) for word in sequence])

    return encoding


def one_hot_decode_sequence(sequence, idx_to_word):
    """One-hot decode sequence.

    :param sequence: One-hot encoded sequence.
    :type sequence: list or :class:`numpy.ndarray`

    :param idx_to_word: Converter with index as key and word as value.
    :type idx_to_word: dict[int, str or int or float]

    :return: One-hot decoded sequence.
    :rtype: list[str or int or float]
    """
    decoding = [idx_to_word[np.argmax(encoded)] for encoded in sequence]

    return decoding


def encode_dataset(X_data, word_to_idx, vocab_size):
    """One-hot encode a set of sequences.

    :param X_data: Contains sequences.
    :type X_data: :class:`numpy.ndarray`

    :param word_to_idx: Converter with word as key and index as value.
    :type word_to_idx: dict[str or int or float, int]

    :param vocab_size: Number of keys in converter.
    :type vocab_size: int

    :return: One-hot encoded dataset.
    :rtype: list[:class:`numpy.ndarray`]
    """
    X_encoded = []

    # Iterate over sequences
    for i in range(X_data.shape[0]):

        sequence = X_data[i]    # Retrieve sequence

        encoded_sequence = one_hot_encode_sequence(sequence, word_to_idx, vocab_size)

        X_encoded.append(encoded_sequence)    # Append to dataset of encoded sequences

    return X_encoded



def extract_blocks(X_data, sizes, strides):
    """.
    """
    ph, pw = sizes
    sh, sw = strides

    idh = [[i + j for j in range(ph + 1)] for i in range(X_data.shape[1] - ph + 1) if i % sh == 0]
    idw = [[i + j for j in range(pw + 1)] for i in range(X_data.shape[2] - pw + 1) if i % sw == 0]

    blocks = []

    for h in idh:
        hs, he = h[0], h[-1]

        blocks.append([])

        for w in idw:
            ws, we = w[0], w[-1]

            blocks[-1].append(X_data[:, hs:he, ws:we, :])

    blocks = np.array(blocks)

    return blocks


def padding(X_data, padding, forward=True):
    """Image padding.

    :param X_data: Array representing a set of images.
    :type X_data: :class:`numpy.ndarray`

    :param padding: Number of zeros to add in each side of the image.
    :type padding: int

    :param forward: Set to False to remove padding, defaults to `True`.
    :type forward: bool, optional
    """
    if forward:
        # Pad image
        shape = ((0, 0), (padding, padding), (padding, padding), (0, 0))
        X_data = np.pad(X_data, shape, mode='constant', constant_values = (0, 0))

    elif not forward:
        # Remove padding
        X_data = X_data[:, padding:-padding, padding:-padding, :]

    return X_data
