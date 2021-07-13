#EpyNN/nnlive/music_author/prepare_dataset.py
import nnlibs.commons.library as cli

import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import tarfile
import random
import glob
import wget
import os


def clips_music(wav_file):
    """

    """

    # Number of bins for signal digitalization
    N_BINS = 16 # 4-bits ADC converter

    # BINS
    BINS = np.linspace(0, 1, N_BINS, endpoint=False)

    # Retrieve sampling rate (Hz) and data from wav file
    SAMPLING_RATE, data = wavfile.read(wav_file)

    # Compute Time (s)
    TIME = data.shape[0] / SAMPLING_RATE

    # Set Time (s) for clip
    CLIP_TIME = 4

    # Compute number of clips
    N_CLIPS = round(TIME//CLIP_TIME)

    # Number of features describing a sample
    N_FEATURES = SAMPLING_RATE * CLIP_TIME

    # Make clips from data
    clips = [ data[i*N_FEATURES:(i+1)*N_FEATURES] for i in range(N_CLIPS) ]

    for i, clip in enumerate(clips):

        # Re-sampling to avoid memory allocation errors
        clip = clip[::SAMPLING_RATE//100]

        # Normalize data in 0-1 range
        clip = np.divide(clip,32768)

        # Digitize and normalize digits
        clips[i] = np.digitize(clip,bins=BINS) / BINS.shape[0]

    return clips


def prepare_dataset(se_dataset):
    """

    """

    # Download data
    if not os.path.exists('./data'):
        url = 'https://synthase.s3.us-west-2.amazonaws.com/music_author_data.tar'
        fname = wget.download(url)
        tar = tarfile.open(fname)
        tar.extractall('./')
        os.remove(fname)

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

    WAV_FILES = glob.glob('./data/*/*wav')

    # Iterate over WAV_FILES
    for wav_file in WAV_FILES:

        # Retrieve clips
        clips = clips_music(wav_file)

        # Iterate over clips
        for features in clips:

            # Test if features associates with p_label (+)
            if 'true' in wav_file:
                sample = [features,p_label]

            # Test if features associates with p_label (+)
            elif 'false' in wav_file:
                sample = [features,n_label]

            # Append sample to dataset
            dataset.append(sample)

    # Shuffle dataset before split
    random.shuffle(dataset)

    # Default N_SAMPLES
    if N_SAMPLES == None:
        N_SAMPLES = len(dataset)

    # Truncate dataset to N_SAMPLES
    dataset = dataset[:N_SAMPLES]

    # Write dataset on disk
    if dataset_save:
        dataset_path = './datasets/'+dataset_name+'.pickle'
        cli.write_pickle(dataset_path,dataset)

    return dataset


def prepare_unlabeled(N_SAMPLES=1):
    """

    """

    # Initialize unlabeled_dataset
    unlabeled_dataset = []

    WAV_FILES = glob.glob('./data/unlabeled/*wav')

    # Iterate over WAV_FILES
    for wav_file in WAV_FILES:

        # Retrieve clips
        clips = clips_music(wav_file)

        # Iterate over clips
        for features in clips:

            sample = [ features, None ]

            # Append sample to dataset
            unlabeled_dataset.append(sample)

    return unlabeled_dataset
