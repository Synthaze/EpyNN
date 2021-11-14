# EpyNN/epynnlive/captcha_mnist/prepare_dataset.py
# Standard library imports
import tarfile
import random
import gzip
import os

# Related third party imports
import wget
import numpy as np

# Local application/library specific imports
from epynn.commons.logs import process_logs


def download_mnist():
    """Download a subset of the MNIST database.
    """
    data_path = os.path.join('.', 'data')

    if not os.path.exists(data_path):

        # Download @url with wget
        url = 'https://synthase.s3.us-west-2.amazonaws.com/mnist_database.tar'
        fname = wget.download(url)

        # Extract archive
        tar = tarfile.open(fname).extractall('.')
        process_logs('Make: '+fname, level=1)

        # Clean-up
        os.remove(fname)

    return None


def prepare_dataset(N_SAMPLES=100):
    """Prepare a dataset of hand-written digits as images.

    :param N_SAMPLES: Number of MNIST samples to retrieve, defaults to 100.
    :type N_SAMPLES: int

    :return: Set of sample features.
    :rtype: tuple[:class:`numpy.ndarray`]

    :return: Set of single-digit sample label.
    :rtype: tuple[:class:`numpy.ndarray`]
    """
    # Process MNIST images
    img_file = gzip.open('data/train-images-idx3-ubyte.gz')

    header = img_file.read(16)
    image_size = int.from_bytes(header[8:12], byteorder='big')
    buf = img_file.read(image_size * image_size * N_SAMPLES)
    X_features = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    X_features = X_features.reshape(N_SAMPLES, image_size, image_size, 1)

    # Process MNIST labels
    label_file = gzip.open('data/train-labels-idx1-ubyte.gz')

    header = label_file.read(8)
    buf = label_file.read(image_size * image_size * N_SAMPLES)
    Y_label = np.frombuffer(buf, dtype=np.uint8)

    # Prepare X-Y pairwise dataset
    dataset = list(zip(X_features, Y_label))

    # Shuffle dataset
    random.shuffle(dataset)

    # Separate X-Y pairs
    X_features, Y_label = zip(*dataset)

    return X_features, Y_label
