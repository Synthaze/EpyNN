#EpyNN/nnlive/dummy_strings/prepare_dataset.py
import nnlibs.commons.library as cli

import random
import glob
import os


def features_string(N_FEATURES):

    # List of words
    WORDS = ['A','T','G','C']

    # Random choice of words for N_FEATURES iterations
    features = [ random.choice(WORDS) for j in range(N_FEATURES) ]

    return features


def prepare_dataset(se_dataset):
    """
    Prepare dummy dataset with Boolean sample features

    sample = [features,label]

    features is a list of sample features with length N_FEATURES
    label is a one-hot encoded label with length N_LABELS

    dataset = [sample_0,sample_1,...,sample_N]
    """

    # See ./settings.py
    N_SAMPLES = se_dataset['N_SAMPLES']

    # See ./settings.py
    dataset_name = se_dataset['dataset_name']
    dataset_save = se_dataset['dataset_save']

    # Initialize dataset
    dataset = []

    # One-hot encoded positive and negative labels
    p_label = [1,0]
    n_label = [0,1]

    # Number of features describing a sample
    N_FEATURES = 12

    # Iterate over N_SAMPLES
    for i in range(N_SAMPLES):

        features = features_string(N_FEATURES)

        # Test if features associates with p_label (+)
        if features.count('A') > 3:
            sample = [features,p_label]

        # Test if features associates with n_label (-)
        elif features.count('A') != 3:
            sample = [features,n_label]

        # Append sample to dataset
        dataset.append(sample)

    # Shuffle dataset before split
    random.shuffle(dataset)

    # Write dataset on disk
    if dataset_save:
        dataset_path = './datasets/'+dataset_name+'.pickle'
        cli.write_pickle(dataset_path,dataset)

    return dataset
# DOCS_END


def prepare_unlabeled(N_SAMPLES=1):

    # Initialize unlabeled_dataset
    unlabeled_dataset = []

    # Number of features describing a sample
    N_FEATURES = 12

    # Iterate over N_SAMPLES
    for i in range(N_SAMPLES):

        features = features_string(N_FEATURES)

        sample = [ features, None ]

        # Append sample to dataset
        unlabeled_dataset.append(sample)

    return unlabeled_dataset

