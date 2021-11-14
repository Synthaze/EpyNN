# EpyNN/epynnlive/ptm_protein/prepare_dataset.py
# Standard library imports
import tarfile
import random
import os

# Related third party imports
import wget

# Local application/library specific imports
from epynn.commons.library import read_file
from epynn.commons.logs import process_logs


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
        process_logs('Make: ' + fname, level=1)

        # Clean-up
        os.remove(fname)

    return None


def prepare_dataset(N_SAMPLES=100):
    """Prepare a set of labeled peptides.

    :param N_SAMPLES: Number of peptide samples to retrieve, defaults to 100.
    :type N_SAMPLES: int

    :return: Set of peptides.
    :rtype: tuple[list[str]]

    :return: Set of single-digit peptides label.
    :rtype: tuple[int]
    """
    # Single-digit positive and negative labels
    p_label = 0
    n_label = 1

    # Positive data are Homo sapiens O-GlcNAcylated peptide sequences from oglcnac.mcw.edu
    path_positive = os.path.join('data', '21_positive.dat')

    # Negative data are peptide sequences presumably not O-GlcNAcylated
    path_negative = os.path.join('data', '21_negative.dat')

    # Read text files, each containing one sequence per line
    positive = [[list(x), p_label] for x in read_file(path_positive).splitlines()]
    negative = [[list(x), n_label] for x in read_file(path_negative).splitlines()]

    # Shuffle data to prevent from any sorting previously applied
    random.shuffle(positive)
    random.shuffle(negative)

    # Truncate to prepare a balanced dataset
    negative = negative[:len(positive)]

    # Prepare a balanced dataset
    dataset = positive + negative

    # Shuffle dataset
    random.shuffle(dataset)

    # Truncate dataset to N_SAMPLES
    dataset = dataset[:N_SAMPLES]

    # Separate X-Y pairs
    X_features, Y_label = zip(*dataset)

    return X_features, Y_label
