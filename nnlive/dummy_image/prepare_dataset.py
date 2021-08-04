# EpyNN/nnlive/dummy_image/prepare_dataset.py
# Standard library imports
import random
import os

# Related third party imports
import matplotlib.pyplot as plt
import numpy as np

# Local application/library specific imports
from nnlibs.commons.library import write_pickle


def features_images(WIDTH=28, HEIGHT=28):
    """Generate dummy image features.

    :param WIDTH: Image width
    :type WIDTH: int

    :param HEIGHT: Image height
    :type HEIGHT: int

    :return: alterated random image features of length N_FEATURES
    :rtype: int

    :return: Value of tone used to alterate random image features
    :rtype: int

    :return: original random image features of length N_FEATURES
    :rtype: list[int]
    """
    # Number of distinct tones in features
    N_TONES = 16

    # Number of channels is one for greyscale images
    DEPTH = 1

    # Number of features describing a sample
    N_FEATURES = WIDTH * HEIGHT

    # Number of channels is one for greyscale images
    DEPTH = 1

    # Shade of greys color palette
    GSCALE = [i for i in range(N_TONES)]

    # Random choice of shade of greys for N_FEATURES iterations
    features = [random.choice(GSCALE) for j in range(N_FEATURES)]
    o_features = features.copy()

    # Random choice darkest or lightest shade in GSCALE
    tone = random.choice([0, N_TONES-1])

    # Random choice of features indexes for 5% of N_FEATURES iterations
    idxs = [random.choice(range(N_FEATURES)) for j in range(N_FEATURES // 20)]

    # Alteration of features with darkest or lightest tone
    for idx in idxs:
        features[idx] = tone

    # Vectorization of features to image and scaling
    features = np.array(features).reshape(HEIGHT, WIDTH, DEPTH) / (N_TONES-1)
    o_features = np.array(o_features).reshape(features.shape) / (N_TONES-1)

    return features, tone, o_features


def label_features(features, tone):
    """Prepare label associated with features.

    :param features: random image features of length N_FEATURES
    :type features: list[int]

    :param tone: Darker or lighter tone used to alterate image features
    :type tone: int

    :return: One-hot encoded label
    :rtype: list[int]
    """
    # One-hot encoded positive and negative labels
    p_label = [1, 0]
    n_label = [0, 1]

    # Test if image was alterated with lighest tone (+)
    if tone == 0:
        label = p_label

    # Test if features associates with darkest tone (-)
    else:
        label = n_label

    return label


def labeled_dataset(se_dataset):
    """Prepare a dummy dataset of labeled samples.

    One sample is a list such as [features, label].

    For one sample, features is a class:`numpy.ndarray` and label is a list.

    :param se_dataset: Settings for dataset preparation
    :type se_dataset: dict

    :return: A dataset of length N_SAMPLES
    :rtype: list[list[class:`numpy.ndarray`,list[int]]]
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
        features, tone, _ = features_images()

        # Retrieve label associated with features
        label = label_features(features, tone)

        # Define labeled sample
        sample = [features, label]

        # Append sample to dataset
        dataset.append(sample)

    # Shuffle dataset
    random.shuffle(dataset)

    # Write dataset on disk
    if dataset_save:
        dataset_path = os.path.join(os.getcwd(), 'datasets', dataset_name+'.pickle')
        write_pickle(dataset_path, dataset)

    return dataset


def unlabeled_dataset(N_SAMPLES=1):
    """Prepare a dummy dataset of unlabeled samples.

    One sample is a list such as [features, []].

    For one sample, features is a class:`numpy.ndarray` and label is an empty list.

    :param N_SAMPLES: Length for unlabeled dataset
    :type N_SAMPLES: int

    :return: A dataset of length N_SAMPLES
    :rtype: list[list[class:`numpy.ndarray`,list[int]]]
    """
    # Initialize unlabeled_dataset
    unlabeled_dataset = []

    # Iterate over N_SAMPLES
    for i in range(N_SAMPLES):

        # Generate dummy image features
        features, _, _ = features_images()

        # Define unlabeled sample
        sample = [features, []]

        # Append to unlabeled_dataset
        unlabeled_dataset.append(sample)

    return unlabeled_dataset
