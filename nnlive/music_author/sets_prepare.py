#EpyNN/nnlive/dummy_wave/sets_prepare.py
import nnlibs.commons.library as cl
import nnlibs.commons.io as cio

from scipy.io import wavfile

import matplotlib.pyplot as plt
import numpy as np

import tarfile
import random
import glob
import wget
import os



def sets_prepare(runData,init=False):

    if not os.path.exists('./data'):
        url = 'https://synthase.s3.us-west-2.amazonaws.com/music_author_data.tar'
        fname = wget.download(url)
        tar = tarfile.open(fname)
        tar.extractall('./')
        os.remove(fname)

    if init == True:
        wav2np()

    positive = []

    for fname in glob.glob('./data/true/*pickle'):

        chunks = cl.read_pickle(fname)['chunks']

        for x in chunks:

            positive.append([x,[1,0]])

    negative = []

    for fname in glob.glob('./data/false/*pickle'):

        chunks = cl.read_pickle(fname)['chunks']

        for x in chunks:

            negative.append([x,[0,1]])

    random.shuffle(positive)
    random.shuffle(negative)

    positive = positive[:len(negative)]

    dataset = positive + negative

    random.shuffle(dataset)

    dsets = cio.dataset_to_dsets(dataset,runData,prefix='dummy_wave',encode=False)

    return dsets


def wav2np():

    for fname in glob.glob('./data/*/*wav'):

        name = fname.split('.wav')[0]

        sr, data = wavfile.read(fname)

        print (data.shape,name)

        print (sr)

        l = data.shape[0] / sr

        norm_data = np.divide(data,32768)

        lc = 4

        nc = round(l//lc)

        chunks = [ norm_data[c*sr*lc:(c+1)*sr*lc] for c in range(nc) ]

        chunks = np.array(chunks)

        data = { 'sr': sr, 'nc': nc, 'chunks': chunks }

        out = name+'.pickle'

        cl.write_pickle(out,data)

    return None
