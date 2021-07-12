#EpyNN/nnlive/dummy_time/prepare_dataset.py
import nnlibs.commons.library as cli
import nnlibs.commons.io as cio

import matplotlib.pyplot as plt
import numpy as np
import random
import glob
import os


def features_time():
    """

    """

    # Number of bins for signal digitalization
    N_BINS = 16 # 4-bits ADC converter

    # Sampling rate (Hz) and Time (s)
    SAMPLING_RATE = 128
    TIME = 1

    # Number of features describing a sample
    N_FEATURES = SAMPLING_RATE * TIME

    # BINS
    BINS = np.linspace(0, 1, N_BINS, endpoint=False)

    # Initialize features array
    raw_features = np.linspace(0, TIME, N_FEATURES, endpoint=False)
    # Random choice of true signal frequency
    signal_frequency = random.uniform(0,SAMPLING_RATE//2)
    # Generate pure sine wave of N_FEATURES points
    raw_features = np.sin(2 * np.pi * signal_frequency * raw_features)

    # Generate white noise
    white_noise = np.random.normal(0, 1, size=N_FEATURES) * 0.1

    raw_features = random.choice([raw_features + white_noise, white_noise])

    features = raw_features + np.abs(np.min(raw_features))
    features /= np.max(features)

    features = np.digitize(features,bins=BINS) / BINS.shape[0]

    return features, raw_features, white_noise


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

    # Iterate over N_SAMPLES
    for i in range(N_SAMPLES):

        features, raw_features, white_noise = features_time()

        # Test if features associates with p_label (+)
        if np.sum(raw_features) != np.sum(white_noise):
            sample = [features,p_label]

        # Test if features associates with n_label (-)
        elif np.sum(raw_features) == np.sum(white_noise):
            sample = [features,n_label]

        # Append sample to dataset
        dataset.append(sample)

    # Plot last features
    plt.plot(features)
    plt.show()

    # Write dataset on disk
    if dataset_save:
        dataset_path = './datasets/'+dataset_name+'.pickle'
        cli.write_pickle(dataset_path,dataset)

    return dataset
# DOCS_END


def prepare_unlabeled(N_SAMPLES=1):
    """

    """

    # Initialize unlabeled_dataset
    unlabeled_dataset = []

    # Iterate over N_SAMPLES
    for i in range(N_SAMPLES):

        features, _, _ = features_time()

        sample = [ features, None ]

        # Append sample to dataset
        unlabeled_dataset.append(sample)

    return unlabeled_dataset
