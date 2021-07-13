#EpyNN/nnlive/mnist_database/prepare_dataset.py
import nnlibs.commons.io as cio

import numpy as np
import tarfile
import random
import wget
import gzip
import os


def prepare_dataset(se_dataset):
    """

    """

    # Download data
    if not os.path.exists('./data'):
        url = 'https://synthase.s3.us-west-2.amazonaws.com/mnist_database_data.tar'
        fname = wget.download(url)
        tar = tarfile.open(fname)
        tar.extractall('./')
        os.remove(fname)

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
    dataset = [ [x,y] for x,y in zip(images,labels) ]

    # Shuffle dataset before split
    random.shuffle(dataset)

    # Default N_SAMPLES
    if N_SAMPLES == None:
        N_SAMPLES = len(dataset)

    # Truncate dataset to N_SAMPLES
    dataset = dataset[:N_SAMPLES]

    return dataset
