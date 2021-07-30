# EpyNN/nnlive/ptm_protein/prepare_dataset.py
# Standard library imports
import tarfile
import random
import wget
import os

# Local application/library specific imports
from nnlibs.commons.library import read_file


def download_sequences():
    """Download a set of peptide sequences.
    """
    data_path = os.path.join('.', 'data')

    if not os.path.exists(data_path):

        # Download @url with wget
        url = 'https://synthase.s3.us-west-2.amazonaws.com/ptm_prediction_data.tar'
        fname = wget.download(url)

        # Extract archive
        tar = tarfile.open(fname).extractall('.')
        process_logs('Make: '+fname, level=1)

        # Clean-up
        os.remove(fname)

    return None


def labeled_dataset(se_dataset):
    """Prepare a dataset of labeled samples.

    One sample is a list such as [features, label].

    For one sample, features is a list and label is a list.

    :param se_dataset: Settings for dataset preparation
    :type se_dataset: dict

    :return: A dataset of length N_SAMPLES
    :rtype: list[list[list[str],list[int]]]
    """
    # See ./settings.py
    N_SAMPLES = se_dataset['N_SAMPLES']

    # One-hot encoded positive and negative labels
    p_label = [1, 0]
    n_label = [0, 1]

    # Positive data are Homo sapiens O-GlcNAcylated peptide sequences from oglcnac.mcw.edu
    path_positive = 'data/21_positive.dat'
    # Negative data are peptide sequences presumably not O-GlcNAcylated
    path_negative = 'data/21_negative.dat'

    # Read text files, each containing one sequence per line
    positive = [[x, p_label] for x in read_file(path_positive).splitlines()]
    negative = [[x, n_label] for x in read_file(path_negative).splitlines()]

    # Shuffle data to prevent from any sorting previously applied
    random.shuffle(positive)
    random.shuffle(negative)

    # Truncate list of negative sequences
    negative = negative[:len(positive)]

    # Prepare a balanced dataset
    dataset = positive + negative

    # Shuffle dataset before split
    random.shuffle(dataset)

    # Truncate dataset to N_SAMPLES
    dataset = dataset[:N_SAMPLES]

    return dataset
