# EpyNN/nnlive/dummy_boolean/prepare_dataset.py
# Standard library imports
import random
import os

# Local application/library specific imports
from nnlibs.commons.library import write_pickle


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

    :return: One-hot encoded label
    :rtype: list[int]
    """
    # One-hot encoded positive and negative labels
    p_label = [1, 0]
    n_label = [0, 1]

    # Test if features contains more True (+)
    if features.count(True) > features.count(False):
        label = p_label

    # Test if features contains more False (-)
    elif features.count(True) < features.count(False):
        label = n_label

    return label


def labeled_dataset(se_dataset):
    """Prepare a dummy dataset of labeled samples.

    One sample is a list such as [features, label].

    For one sample, features is a list and label is a list.

    :param se_dataset: Settings for dataset preparation
    :type se_dataset: dict

    :return: A dataset of length N_SAMPLES
    :rtype: list[list[list[bool],list[int]]]
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

        # Compute random boolean features
        features = features_boolean()

        # Retrieve label associated with features
        label = label_features(features)

        # Define labeled sample
        sample = [features, label]

        # Append sample to dataset
        dataset.append(sample)

    # Shuffle dataset
    random.shuffle(dataset)

    # Write dataset on disk
    if dataset_save:
        dataset_path = os.path.join(os.getcwd(), 'dataset', dataset_name+'.pickle')
        write_pickle(dataset_path,dataset)

    return dataset


def unlabeled_dataset(N_SAMPLES=1):
    """Prepare a dummy dataset of unlabeled samples.

    One sample is a list such as [features, []].

    For one sample, features is a list and label is an empty list.

    :param N_SAMPLES: Length for unlabeled dataset
    :type N_SAMPLES: int

    :return: A dataset of length N_SAMPLES
    :rtype: list[list[list[bool],list]]
    """
    # Initialize unlabeled_dataset
    unlabeled_dataset = []

    # Iterate over N_SAMPLES
    for i in range(N_SAMPLES):

        # Generate dummy boolean features
        features = features_boolean()

        # Define unlabeled sample
        sample = [features, []]

        # Append to unlabeled_dataset
        unlabeled_dataset.append(sample)

    return unlabeled_dataset
