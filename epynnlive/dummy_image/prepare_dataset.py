# EpyNN/epynnlive/dummy_image/prepare_dataset.py
# Standard library imports
import random

# Related third party imports
import numpy as np


def features_image(WIDTH=28, HEIGHT=28):
    """Generate dummy image features.

    :param WIDTH: Image width, defaults to 28.
    :type WIDTH: int

    :param HEIGHT: Image height, defaults to 28.
    :type HEIGHT: int

    :return: Random image features of size N_FEATURES.
    :rtype: :class:`numpy.ndarray`

    :return: Non-random image features of size N_FEATURES.
    :rtype: :class:`numpy.ndarray`
    """
    # Number of channels is one for greyscale images
    DEPTH = 1

    # Number of features describing a sample
    N_FEATURES = WIDTH * HEIGHT * DEPTH

    # Number of distinct tones in features
    N_TONES = 16

    # Shades of grey
    GSCALE = [i for i in range(N_TONES)]

    # Random choice of shades for N_FEATURES iterations
    features = [random.choice(GSCALE) for j in range(N_FEATURES)]

    # Vectorization of features
    features = np.array(features).reshape(HEIGHT, WIDTH, DEPTH)

    # Masked features
    mask_on_features = features.copy()
    mask_on_features[np.random.randint(0, HEIGHT)] = np.zeros_like(features[0])
    mask_on_features[:, np.random.randint(0, WIDTH)] = np.zeros_like(features[:, 0])

    # Random choice between random image or masked image
    features = random.choice([features, mask_on_features])

    return features, mask_on_features


def label_features(features, mask_on_features):
    """Prepare label associated with features.

    The dummy law is:

    Image is NOT random = positive
    Image is random = negative

    :param features: Random image features of size N_FEATURES
    :type features: :class:`numpy.ndarray`

    :param mask_on_features: Non-random image features of size N_FEATURES
    :type mask_on_features: :class:`numpy.ndarray`

    :return: Single-digit label with respect to features
    :rtype: int
    """
    # Single-digit positive and negative labels
    p_label = 0
    n_label = 1

    # Test if image is not random (0)
    if np.sum(features) == np.sum(mask_on_features):
        label = p_label

    # Test if image is random (1)
    elif np.sum(features) != np.sum(mask_on_features):
        label = n_label

    return label


def prepare_dataset(N_SAMPLES=100):
    """Prepare a set of dummy time sample features and label.

    :param N_SAMPLES: Number of samples to generate, defaults to 100.
    :type N_SAMPLES: int

    :return: Set of sample features.
    :rtype: tuple[:class:`numpy.ndarray`]

    :return: Set of single-digit sample label.
    :rtype: tuple[int]
    """
    # Initialize X and Y datasets
    X_features = []
    Y_label = []

   # Iterate over N_SAMPLES
    for i in range(N_SAMPLES):

        # Compute random string features
        features, mask_on_features = features_image()

        # Retrieve label associated with features
        label = label_features(features, mask_on_features)

        # Append sample features to X_features
        X_features.append(features)

        # Append sample label to Y_label
        Y_label.append(label)

    # Prepare X-Y pairwise dataset
    dataset = list(zip(X_features, Y_label))

    # Shuffle dataset
    random.shuffle(dataset)

    # Separate X-Y pairs
    X_features, Y_label = zip(*dataset)

    return X_features, Y_label
