#EpyNN/nnlive/dummy_images/prepare_dataset.py
import nnlibs.commons.library as cli
import nnlibs.commons.io as cio

import matplotlib.pyplot as plt
import numpy as np
import random
import glob
import os


def prepare_dataset(se_dataset):
    """
    Prepare dummy dataset with Boolean sample features

    sample = [features,label]

    features is a list of sample features with length N_FEATURES
    label is a one-hot encoded label with length N_LABELS

    dataset = [sample_0,sample_1,...,sample_N]
    """

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

    # Number of distinct tones in features
    N_TONES = 16

    # Shade of greys color palette
    GSCALE = [ i for i in range(N_TONES) ]

    # Number of channels is one for greyscale images
    DEPTH = 1

    # Image dimensions
    WIDTH = 28
    HEIGHT = 16

    # Number of features describing a sample
    N_FEATURES = WIDTH * HEIGHT

    # Iterate over N_SAMPLES
    for i in range(N_SAMPLES):
        # Random choice of shade of greys for N_FEATURES iterations
        features = [ random.choice(GSCALE) for j in range(N_FEATURES) ]

        # Random choice darkest or lightest shade in GSCALE
        tone = random.choice([0,N_TONES-1])
        # Random choice of features indexes for 5% of N_FEATURES iterations
        idxs = [ random.choice(range(N_FEATURES)) for j in range(N_FEATURES//20) ]

        # Alteration of features with darkest or lightest tone
        for idx in idxs:
            features[idx] = tone

        # Vectorization of features to image
        image = np.array(features).reshape(HEIGHT,WIDTH,DEPTH)

        # Test if features associates with p_label (+)
        if tone == 0:
            sample = [image,p_label]

        # Test if features associates with n_label (-)
        elif tone == N_TONES - 1:
            sample = [image,n_label]

        # Append sample to dataset
        dataset.append(sample)

    # Plot last image for axis 0 (HEIGHT) and 1 (WIDTH)
    plt.imshow(image[:,:,0], cmap='gray', vmin=0, vmax=N_TONES-1)
    # Display last image
    plt.show()

    # Write dataset on disk
    if dataset_save:
        dataset_path = './datasets/'+dataset_name+'.pickle'
        cli.write_pickle(dataset_path,dataset)

    return dataset
# DOCS_END


def read_dataset(dataset_path=None):

    if dataset_path == None:
        dataset_path = max(glob.glob('./datasets/*'), key=os.path.getctime)

    dataset = cli.read_pickle(dataset_path)

    return dataset
