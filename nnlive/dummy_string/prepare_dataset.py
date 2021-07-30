# EpyNN/nnlive/dummy_strings/prepare_dataset.py
# Standard library imports
import random
import os

# Local application/library specific imports
from nnlibs.commons.library import write_pickle


def features_string():
    """Generate dummy string features.

    :return: Words that may be contained in features
    :rtype: list[str]

    :return: random string features of length N_FEATURES
    :rtype: list[str]
    """
    # Number of features
    N_FEATURES = 12

    # List of words
    WORDS = ['A', 'T', 'G', 'C']

    # Random choice of words for N_FEATURES iterations
    features = [random.choice(WORDS) for j in range(N_FEATURES)]

    return features, WORDS


def label_features(features, WORDS):
    """Prepare label associated with features.

    :param features: random string features of length N_FEATURES
    :type features: list[str]

    :param WORDS: Words that may be contained in features
    :type WORDS: list[str]

    :return: One-hot encoded label
    :rtype: list[int]
    """
    # One-hot encoded positive and negative labels
    p_label = [1, 0]
    n_label = [0, 1]

    # Number of features
    N_FEATURES = len(features)

    # Mean distribution for words in features
    mean_distribution = N_FEATURES // len(WORDS)

    # Target word
    adenosine = WORDS[0]

    # Test if word count deviates from mean (+)
    if features.count(adenosine) == mean_distribution:
            label = p_label

    # Test if word count does not deviate from mean (-)
    elif features.count(adenosine) != mean_distribution:
            label = n_label

    return label


def labeled_dataset(se_dataset):
    """Prepare a dummy dataset of labeled samples.

    One sample is a list such as [features, label].

    For one sample, features is a list and label is a list.

    :param se_dataset: Settings for dataset preparation
    :type se_dataset: dict

    :return: A dataset of length N_SAMPLES
    :rtype: list[list[list[str],list[int]]]
    """
    # See ./settings.py
    N_SAMPLES = se_dataset['N_SAMPLES']

    # See ./settings.py
    dataset_name = se_dataset['dataset_name']
    dataset_save = se_dataset['dataset_save']

    # Initialize dataset
    dataset = []

    # Iterate over N_SAMPLES
    for i in range(N_SAMPLES):

        # Generate dummy string features
        features, WORDS = features_string()

        # Retrieve label associated with features
        label = label_features(features, WORDS)

        # Define labeled sample
        sample = [features, label]

        # Append sample to dataset
        dataset.append(sample)

    # Shuffle dataset
    random.shuffle(dataset)

    # Write dataset on disk
    if dataset_save:
        dataset_path = os.path.join(os.getcwd(), 'dataset', dataset_name+'.pickle')
        write_pickle(dataset_path, dataset)

    return dataset


def unlabeled_dataset(N_SAMPLES=1):
    """Prepare a dummy dataset of unlabeled samples.

    One sample is a list such as [features, []].

    For one sample, features is a list and label is an empty list.

    :param N_SAMPLES: Length for unlabeled dataset
    :type N_SAMPLES: int

    :return: A dataset of length N_SAMPLES
    :rtype: list[list[list[str],list]]
    """
    # Initialize unlabeled_dataset
    unlabeled_dataset = []

    # Iterate over N_SAMPLES
    for i in range(N_SAMPLES):

        # Generate dummy string features
        features, _ = features_string()

        # Define unlabeled sample
        sample = [features, []]

        # Append sample to dataset
        unlabeled_dataset.append(sample)

    return unlabeled_dataset
