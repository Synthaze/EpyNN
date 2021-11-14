# EpyNN/epynn/commons/io.py
# Related third party imports
import numpy as np

 
def index_elements_auto(X_data):
    """Determine elements size and generate dictionary for one-hot encoding or features or label.

    :param X_data: Dataset containing samples features or samples label.
    :type X_data: :class:`numpy.ndarray`

    :return: One-hot encoding converter.
    :rtype: dict[str or int or float, int]

    :return: One-hot decoding converter.
    :rtype: dict[int, str or int or float]

    :return: Vocabulary size.
    :rtype: int
    """
    X_data = X_data.flatten().tolist()       # All elements in 1D list

    elements = sorted(list(set(X_data)))     # Unique elements list
    elements_size = len(elements)            # Number of elements

    # Converters to encode and decode sequences
    element_to_idx = {w: i for i, w in enumerate(elements)}
    idx_to_element = {i: w for w, i in element_to_idx.items()}

    return element_to_idx, idx_to_element, elements_size


def scale_features(X_data):
    """Scale input array within [0, 1].

    :param X_data: Raw data.
    :type X_data: :class:`numpy.ndarray`

    :return: Normalized data.
    :rtype: :class:`numpy.ndarray`
    """
    X_data = (X_data-np.min(X_data)) / (np.max(X_data)-np.min(X_data))

    return X_data


def one_hot_encode(i, elements_size):
    """Generate one-hot encoding array.

    :param i: One-hot index for current word.
    :type i: int

    :param elements_size: Number of keys in the word to index encoder.
    :type elements_size: int

    :return: One-hot encoding array for current word.
    :rtype: :class:`numpy.ndarray`
    """
    one_hot = np.zeros(elements_size)

    one_hot[i] = 1.0    # Set 1 at index assigned to word

    return one_hot


def one_hot_encode_sequence(sequence, element_to_idx, elements_size):
    """One-hot encode sequence.

    :param sequence: Sequential data.
    :type sequence: list or :class:`numpy.ndarray`

    :param element_to_idx: Converter with word as key and index as value.
    :type element_to_idx: dict[str or int or float, int]

    :param elements_size: Number of keys in converter.
    :type elements_size: int

    :return: One-hot encoded sequence.
    :rtype: :class:`numpy.ndarray`
    """
    encoding = np.array([one_hot_encode(element_to_idx[word], elements_size) for word in sequence])

    return encoding


def one_hot_decode_sequence(sequence, idx_to_element):
    """One-hot decode sequence.

    :param sequence: One-hot encoded sequence.
    :type sequence: list or :class:`numpy.ndarray`

    :param idx_to_element: Converter with index as key and word as value.
    :type idx_to_element: dict[int, str or int or float]

    :return: One-hot decoded sequence.
    :rtype: list[str or int or float]
    """
    decoding = [idx_to_element[np.argmax(encoded)] for encoded in sequence]

    return decoding


def encode_dataset(X_data, element_to_idx, elements_size):
    """One-hot encode a set of sequences.

    :param X_data: Contains sequences.
    :type X_data: :class:`numpy.ndarray`

    :param element_to_idx: Converter with word as key and index as value.
    :type element_to_idx: dict[str or int or float, int]

    :param elements_size: Number of keys in converter.
    :type elements_size: int

    :return: One-hot encoded dataset.
    :rtype: list[:class:`numpy.ndarray`]
    """
    X_encoded = []

    # Iterate over sequences
    for i in range(X_data.shape[0]):

        sequence = X_data[i]    # Retrieve sequence

        encoded_sequence = one_hot_encode_sequence(sequence, element_to_idx, elements_size)

        X_encoded.append(encoded_sequence)    # Append to dataset of encoded sequences

    return X_encoded


def padding(X_data, padding, forward=True):
    """Image padding.

    :param X_data: Array representing a set of images.
    :type X_data: :class:`numpy.ndarray`

    :param padding: Number of zeros to add in each side of the image.
    :type padding: int

    :param forward: Set to False to remove padding, defaults to `True`.
    :type forward: bool, optional
    """
    if padding and forward:
        # Pad image
        shape = ((0, 0), (padding, padding), (padding, padding), (0, 0))
        X_data = np.pad(X_data, shape, mode='constant', constant_values=(0, 0))

    elif padding and not forward:
        # Remove padding
        X_data = X_data[:, padding:-padding, padding:-padding, :]

    return X_data
