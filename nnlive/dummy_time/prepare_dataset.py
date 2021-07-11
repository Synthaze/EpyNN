#EpyNN/nnlive/dummy_time/prepare_dataset.py
import nnlibs.commons.library as cli
import nnlibs.commons.io as cio

import matplotlib.pyplot as plt
import numpy as np
import random
import glob
import os


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

    # Number of bins for signal digitalization
    N_BINS = 128 # 4-bits ADC converter

    # BINS
    BINS = np.linspace(0, 1, N_BINS, endpoint=False)

    # Sampling rate (Hz) and Time (s)
    SAMPLING_RATE = 64
    TIME = 4

    # Number of features describing a sample
    N_FEATURES = SAMPLING_RATE * TIME

    # Iterate over N_SAMPLES
    for i in range(N_SAMPLES):
        # Initialize features array
        features = np.linspace(0, TIME, N_FEATURES, endpoint=False)
        # Random choice of true signal frequency
        signal_frequency = random.uniform(0,SAMPLING_RATE//2)
        # Generate pure sine wave of N_FEATURES points
        features = np.sin(2 * np.pi * signal_frequency * features)

        # Generate white noise
        white_noise = np.random.normal(0, 1, size=N_FEATURES) * 0.1

        signal = features = random.choice([features + white_noise, white_noise])

        signal = signal + np.abs(np.min(features))
        signal /= np.max(signal)

        signal = np.digitize(signal,bins=BINS)

        # Test if features associates with p_label (+)
        if np.sum(features) != np.sum(white_noise):
            sample = [signal,p_label]

        # Test if features associates with n_label (-)
        elif np.sum(features) == np.sum(white_noise):
            sample = [signal,n_label]

        # Append sample to dataset
        dataset.append(sample)

    # Plot last signal
    plt.plot(signal)
    plt.show()

    # Write dataset on disk
    if dataset_save:
        dataset_path = './datasets/'+dataset_name+'.pickle'
        cli.write_pickle(dataset_path,dataset)

    return dataset
# DOCS_END


def read_dataset(dataset_path=None):

    if dataset_path == None:
        dataset_path = max(glob.glob('./datasets/*'), key=os.path.getctime)

    dataset = cli.read_pickle(dataset_path)

    return dataset
