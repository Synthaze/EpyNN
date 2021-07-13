#EpyNN/nnlive/ptm_prediction/prepare_dataset.py
import nnlibs.commons.library as cl

import tarfile
import random
import wget
import os


def prepare_dataset(se_dataset):
    """

    """

    # Download data
    if not os.path.exists('./data'):
        url = 'https://synthase.s3.us-west-2.amazonaws.com/ptm_prediction_data.tar'
        fname = wget.download(url)
        tar = tarfile.open(fname)
        tar.extractall('./')
        os.remove(fname)

    # See ./settings.py
    N_SAMPLES = se_dataset['N_SAMPLES']

    # See ./settings.py
    dataset_name = se_dataset['dataset_name']
    dataset_save = se_dataset['dataset_save']

    # One-hot encoded positive and negative labels
    p_label = [1,0]
    n_label = [0,1]

    # Positive data are Homo sapiens O-GlcNAcylated peptide sequences from oglcnac.mcw.edu
    path_positive = 'data/21_positive.dat'
    # Negative data are peptide sequences presumably not O-GlcNAcylated
    path_negative = 'data/21_negative.dat'

    # Read text files, each containing one sequence per line
    positive = [ [x,p_label] for x in cl.read_file(path_positive).splitlines() ]
    negative = [ [x,n_label] for x in cl.read_file(path_negative).splitlines() ]

    # Shuffle data to prevent from any sorting previously applied
    random.shuffle(positive)
    random.shuffle(negative)

    # Truncate list of negative sequences
    negative = negative[:len(positive)]

    # Prepare a balanced dataset
    dataset = positive + negative

    # Shuffle dataset before split
    random.shuffle(dataset)

    # Default N_SAMPLES
    if N_SAMPLES == None:
        N_SAMPLES = len(dataset)

    # Truncate dataset to N_SAMPLES
    dataset = dataset[:N_SAMPLES]

    return dataset


def prepare_unlabeled(N_SAMPLES=1):
    """

    """

    # Initialize unlabeled_dataset
    unlabeled_dataset = []

    path_unlabeled = 'data/21_unlabeled.dat'

    # Read text files, each containing one sequence per line
    unlabeled_dataset = [ [x,None] for x in cl.read_file(path_unlabeled).splitlines() ]

    return unlabeled_dataset
