#EpyNN/nnlive/dummy_wave/sets_prepare.py
import nnlibs.commons.library as cl
import nnlibs.commons.io as cio

from scipy.io import wavfile

import matplotlib.pyplot as plt
import numpy as np

import random
import glob


def sets_prepare(runData):

    fs = 44100

    t = 4

    samples = np.linspace(0, t, int(fs*t), endpoint=False)

    f_positive = 16

    positive = []

    for i in range(100):
        f_positive += random.uniform(-1,1)
        x = np.sin(2 * np.pi * f_positive * samples)
        positive.append([x,[1,0]])

    f_negative = 64

    negative = []

    for i in range(100):
        f_negative += random.uniform(-1,1)
        x = np.sin(2 * np.pi * f_negative * samples)
        negative.append([x,[0,1]])

    dataset = positive + negative

    random.shuffle(dataset)

    dsets = cio.dataset_to_dsets(dataset,runData,prefix='dummy_wave',encode=False)

    return dsets


def unk_prepare():

    fs = 44100

    t = 4

    samples = np.linspace(0, t, int(fs*t), endpoint=False)

    data = []

    freqs = []

    for i in range(10):

        f_data = [16,64][random.randint(0,1)]

        x = np.sin(2 * np.pi * f_data * samples)

        data.append(x)

        freqs.append(f_data)

    return np.array(data), freqs
