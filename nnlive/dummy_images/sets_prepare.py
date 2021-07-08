#EpyNN/nnlive/mnist_database/sets_prepare.py
import nnlibs.commons.io as cio

import numpy as np
import tarfile
import random
import wget
import gzip
import os

import matplotlib.pyplot as plt

from PIL import Image

def sets_prepare(runData):

    WIDTH = 28

    HEIGHT = 28

    PIXELS = WIDTH * HEIGHT

    N_TONES = 16



    BW_COLORS = [ random.randint(0,255) for i in range(N_TONES) ]

    N_SAMPLES = runData.m['s']

    dataset = []

    dataset = [ np.random.randint(0,256,PIXELS) for i in range(N_SAMPLES) ]

    positive = []
    negative = []

    for i in range(N_SAMPLES):

        tone = random.choice([0,255])

        image = dataset[i].copy()

        for j in range(PIXELS//2):

            idx = random.randint(0,PIXELS-1)

            image[idx] = tone

        image.resize((HEIGHT,WIDTH,1))

        if tone == 255:
            positive.append(image.copy())

        elif tone == 0:
            negative.append(image.copy())

    # Positive features associates with positive label [1,0]
    positive = [ [f,[1,0]] for f in positive ]

    # Negative features associates with negative label [0,1]
    negative = [ [f,[0,1]] for f in negative ]

    # Concatenate posive and negative lists of samples
    dataset = positive + negative

    # Shuffle dataset before split
    random.shuffle(dataset)

#    plt.imshow(positive[0][:,:,0], cmap='gray', vmin=0, vmax=255)

#    plt.show()


    # Return dsets = [dtrain,dtest,dval]
    dsets = cio.dataset_to_dsets(dataset,runData,prefix='dummy_images')

    return dsets
