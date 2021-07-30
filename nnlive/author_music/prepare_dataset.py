# EpyNN/nnlive/author_music/prepare_dataset.py
# Standard library imports
import tarfile
import random
import glob
import wget
import os

# Related third party imports
import numpy as np
from scipy.io import wavfile

# Local application/library specific imports
from nnlibs.commons.library import write_pickle


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


def clips_music(wav_file, clip_time=4, sampling_step=100, digitize=True):
    """Clip music and proceed with resampling and digitalization.

    :param wav_file: The filename of .wav file which contains the music
    :type wav_file: str

    :param clip_time: Duration of one clip
    :type clip_time: int

    :param sampling_step: Factor by which the original sampling rate is divided
    :type sampling_step: int

    :param digitize: Proceed or not with digitalization
    :type digitize: bool

    :return: A list of clips of duration equal to CLIP_TIME
    :rtype: list[class:`numpy.ndarray`]
    """
    # Number of bins for signal digitalization
    N_BINS = 16 # 4-bits ADC converter

    # BINS
    BINS = np.linspace(0, 1, N_BINS, endpoint=False)

    # Retrieve sampling rate (Hz) and data from wav file
    SAMPLING_RATE, data = wavfile.read(wav_file)

    time = data.shape[0] / SAMPLING_RATE

    if clip_time != None:

        N_CLIPS = int(time / clip_time)
        N_FEATURES = SAMPLING_RATE * clip_time

        # Make clips from data
        clips = [data[i*N_FEATURES:(i+1)*N_FEATURES] for i in range(N_CLIPS)]

    else:
        clips = [data]

    for i, clip in enumerate(clips):

        # Re-sampling to avoid memory allocation errors
        clips[i] = clip[::sampling_step]

        # Normalize data in 0-1 range
        clip = np.divide(clip, 32768)

        if digitize:
            # Digitize and normalize digits
            clips[i] = np.digitize(clip, bins=BINS) / BINS.shape[0]

    return clips


def labeled_dataset(se_dataset):
    """Prepare a dataset of labeled samples.

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

    # One-hot encoded positive and negative labels
    p_label = [1, 0]
    n_label = [0, 1]

    wav_paths = os.path.join('data', '*', '*wav')

    WAV_FILES = glob.glob(wav_paths)

    # Iterate over WAV_FILES
    for wav_file in WAV_FILES:

        # Retrieve clips
        clips = clips_music(wav_file)

        # Iterate over clips
        for features in clips:

            # Test if features is from author A (+)
            if 'true' in wav_file:
                label = p_label

            # Test if features is from author B (-)
            elif 'false' in wav_file:
                label = n_label

            # Define labeled sample
            sample = [features, label]

            # Append sample to dataset
            dataset.append(sample)

    # Shuffle dataset before split
    random.shuffle(dataset)

    # Truncate dataset to N_SAMPLES
    dataset = dataset[:N_SAMPLES]

    # Write dataset on disk
    if dataset_save:
        dataset_path = './datasets/'+dataset_name+'.pickle'
        cli.write_pickle(dataset_path,dataset)

    return dataset
