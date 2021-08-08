# EpyNN/nnlive/author_music/prepare_dataset.py
# Standard library imports
import tarfile
import random
import glob
import gzip
import wget
import os

# Related third party imports
import numpy as np
from scipy.io import wavfile

# Local application/library specific imports
from nnlibs.commons.logs import process_logs


def download_music():
    """Download some guitar music.
    """
    data_path = os.path.join('.', 'data')

    if not os.path.exists(data_path):

        # Download @url with wget
        url = 'https://synthase.s3.us-west-2.amazonaws.com/music_author_data.tar'
        fname = wget.download(url)

        # Extract archive
        tar = tarfile.open(fname).extractall('.')
        process_logs('Make: '+fname, level=1)

        # Clean-up
        os.remove(fname)

    return None


def clips_music(wav_file, TIME=4, SAMPLING_RATE=4410):
    """Clip music and proceed with resampling.

    :param wav_file: The filename of .wav file which contains the music.
    :type wav_file: str

    :param SAMPLING_RATE: Sampling rate (Hz), default to 4410.
    :type SAMPLING_RATE: int

    :param TIME: Sampling time (s), defaults to 4.
    :type TIME: int

    :return: Clipped and re-sampled music.
    :rtype: list[:class:`numpy.ndarray`]
    """
    # Number of features describing a sample
    N_FEATURES = SAMPLING_RATE * TIME

    # Retrieve original sampling rate (Hz) and data
    wav_sampling_rate, wav_data = wavfile.read(wav_file)

    # Total duration (s) of original data
    wav_time = wav_data.shape[0] / wav_sampling_rate

    # Number of clips to slice from original data
    N_CLIPS = int(wav_time / TIME)

    # Make clips from data
    clips = [wav_data[i * N_FEATURES:(i+1) * N_FEATURES] for i in range(N_CLIPS)]

    # Compute step for re-sampling
    sampling_step = int(wav_sampling_rate / SAMPLING_RATE)

    # Re-sampling to avoid memory allocation errors
    clips = [clip[::sampling_step] for clip in clips]

    return clips


def prepare_dataset(N_SAMPLES=100):
    """Prepare a dataset of clipped music as NumPy arrays.

    :param N_SAMPLES: Number of clip samples to retrieve, defaults to 100.
    :type N_SAMPLES: int

    :return: Set of sample features.
    :rtype: tuple[:class:`numpy.ndarray`]

    :return: Set of single-digit sample label.
    :rtype: tuple[:class:`numpy.ndarray`]
    """
    # Initialize X and Y datasets
    X_features = []
    Y_label = []

    wav_paths = os.path.join('data', '*', '*wav')

    wav_files = glob.glob(wav_paths)

    # Iterate over WAV_FILES
    for wav_file in wav_files:

        # Retrieve clips
        clips = clips_music(wav_file)

        # Iterate over clips
        for features in clips:

            # Clip is positive if played by true author (+) else negative (-)
            label = 1 if 'true' in wav_file else 0

            # Append sample features to X_features
            X_features.append(features)

            # Append sample label to Y_label
            Y_label.append(label)

    # Prepare X-Y pairwise dataset
    dataset = list(zip(X_features, Y_label))

    # Shuffle dataset
    random.shuffle(dataset)

    # Truncate dataset to N_SAMPLES
    dataset = dataset[:N_SAMPLES]

    # Separate X-Y pairs
    X_features, Y_label = zip(*dataset)

    return X_features, Y_label
