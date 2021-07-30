# EpyNN/nnlive/captcha_mnist/prepare_dataset.py
# Standard library imports
import tarfile
import random
import wget
import gzip
import os

# Related third party imports
import numpy as np

# Local application/library specific imports
from nnlibs.commons.logs import process_logs


def download_mnist():
    """Download a subset of the MNIST database.
    """
    data_path = os.path.join('.', 'data')

    if not os.path.exists(data_path):

        # Download @url with wget
        url = 'https://synthase.s3.us-west-2.amazonaws.com/mnist_database_data.tar'
        fname = wget.download(url)

        # Extract archive
        tar = tarfile.open(fname).extractall('.')
        process_logs('Make: '+fname, level=1)

        # Clean-up
        os.remove(fname)

    return None


def prepare_dataset(se_dataset):
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

    # See ./settings.py
    dataset_name = se_dataset['dataset_name']
    dataset_save = se_dataset['dataset_save']

    # Process MNIST images
    f = gzip.open('data/train-images.gz')

    header = f.read(16)
    num_images = int.from_bytes(header[4:8], byteorder='big')
    image_size = int.from_bytes(header[8:12], byteorder='big')
    buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    images = data.reshape(num_images, image_size, image_size, 1)

    # Process MNIST labels
    f = gzip.open('data/train-labels.gz')

    header = f.read(8)
    num_labels = int.from_bytes(header[4:8], byteorder='big')
    buf = f.read(image_size * image_size * num_images)
    labels = np.frombuffer(buf, dtype=np.uint8)

    num_classes = np.max(labels) + 1

    # one-hot-encoded labels
    labels = np.eye(num_classes)[labels]

    # Initialize dataset
    dataset = [[x,y] for x,y in zip(images,labels)]

    # Shuffle dataset before split
    random.shuffle(dataset)

    # Default N_SAMPLES
    if N_SAMPLES == None:
        N_SAMPLES = len(dataset)

    # Truncate dataset to N_SAMPLES
    dataset = dataset[:N_SAMPLES]

    return dataset
