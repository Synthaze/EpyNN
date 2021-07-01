#EpyNN/nnlive/mnist_database/sets_prepare.py
import nnlibs.commons.io as cio

import numpy as np
import random
import gzip
import os


def sets_prepare(runData):

    os.system('cat data/*part* > data/train-images.gz')

    dataset = []

    f = gzip.open('data/train-images.gz')

    header = f.read(16)
    num_images = int.from_bytes(header[4:8], byteorder='big')
    image_size = int.from_bytes(header[8:12], byteorder='big')
    buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    dataset.append(data.reshape(num_images, image_size, image_size, 1))

    f = gzip.open('data/train-labels.gz')

    header = f.read(8)
    num_labels = int.from_bytes(header[4:8], byteorder='big')
    buf = f.read(image_size * image_size * num_images)
    dataset.append(np.frombuffer(buf, dtype=np.uint8))

    X = dataset[0]
    Y = dataset[1]

    num_classes = np.max(Y) + 1

    Y = np.eye(num_classes)[Y]    # one-hot-encoded labels

    dataset = [ [x,y] for x,y in zip(X,Y) ]

    random.shuffle(dataset)

    if runData.m['s'] == None:
        runData.m['s'] = len(dataset)

    dataset = dataset[:runData.m['s']]

    dsets = cio.dataset_to_dsets(dataset,runData,prefix='mnist')

    return dsets
