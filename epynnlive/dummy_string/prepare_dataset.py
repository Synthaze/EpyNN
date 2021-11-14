# EpyNN/epynnlive/dummy_strings/prepare_dataset.py
# Standard library imports
import random


def features_string(N_FEATURES=12):
    """Generate dummy string features.

    :param N_FEATURES: Number of features, defaults to 12.
    :type N_FEATURES: int

    :return: random string features of length N_FEATURES.
    :rtype: list[str]
    """
    # List of words
    WORDS = ['A', 'T', 'G', 'C']

    # Random choice of words for N_FEATURES iterations
    features = [random.choice(WORDS) for j in range(N_FEATURES)]

    return features


def label_features(features):
    """Prepare label associated with features.

    The dummy law is:

    First and last elements are equal = positive.
    First and last elements are NOT equal = negative.

    :param features: random string features of length N_FEATURES.
    :type features: list[str]

    :return: Single-digit label with respect to features.
    :rtype: int
    """
    # Single-digit positive and negative labels
    p_label = 0
    n_label = 1

    # Pattern associated with positive label (0)
    if features[0] == features[-1]:
            label = p_label

    # Other pattern associated with negative label (1)
    elif features[0] != features[-1]:
            label = n_label

    return label


def prepare_dataset(N_SAMPLES=100):
    """Prepare a set of dummy string sample features and label.

    :param N_SAMPLES: Number of samples to generate, defaults to 100.
    :type N_SAMPLES: int

    :return: Set of sample features.
    :rtype: tuple[list[str]]

    :return: Set of single-digit sample label.
    :rtype: tuple[int]
    """
    # Initialize X and Y datasets
    X_features = []
    Y_label = []

   # Iterate over N_SAMPLES
    for i in range(N_SAMPLES):

        # Compute random string features
        features = features_string()

        # Retrieve label associated with features
        label = label_features(features)

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
