# EpyNN/nnlive/dummy_boolean/prepare_dataset.py
# Standard library imports
import random
import os


def features_boolean(N_FEATURES=11):
    """Generate dummy string features.

    :param N_FEATURES: Number of features
    :type N_FEATURES: int

    :return: random boolean features of length N_FEATURES
    :rtype: list[bool]
    """
    # Random choice True or False for N_FEATURES iterations
    features = [random.choice([True, False]) for j in range(N_FEATURES)]

    return features


def label_features(features):
    """Prepare label associated with features.

    :param features: random boolean features of length N_FEATURES
    :type features: list[bool]

    :return: Single-digit label with respect to features
    :rtype: int
    """
    # Single-digit positive and negative labels
    p_label = 1
    n_label = 0

    # Test if features contains more True (+)
    if features.count(True) > features.count(False):
        label = p_label

    # Test if features contains more False (-)
    elif features.count(True) < features.count(False):
        label = n_label

    return label


def prepare_dataset(N_SAMPLES=100):
    """Prepare a dummy dataset of labeled samples.

    :param N_SAMPLES: Number of samples to generate
    :type N_SAMPLES: int

    :return:
    :rtype:
    """
    # Initialize X and Y datasets
    X_features = []
    Y_label = []

   # Iterate over N_SAMPLES
    for i in range(N_SAMPLES):

        # Compute random boolean features
        features = features_boolean()

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
