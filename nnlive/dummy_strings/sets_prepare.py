#EpyNN/nnlive/dummy_strings/sets_prepare.py
import nnlibs.commons.io as cio

import random


def sets_prepare(runData):
    """
    Prepare dummy dataset with bolean sample features

    sample = [features,label]

    features is a one-hot encoded string of shape (F_LENGTH,vocab_size)
    label is a one-hot encoded label of shape (N_LABELS,)
    """

    dataset = []

    # See settings.py - config dict is cached into runData.m
    N_SAMPLES = runData.m['s']

    # LENGTH is the number of features describing a sample
    F_LENGTH = 11

    positive = []
    negative = []

    # Iterate until positive + negative is equal to N_SAMPLES
    while len(positive) + len(negative) < N_SAMPLES:
        # features is a 1D list of length F_LENGTH containing random boleans
        features = [ ['A','B','C'][random.randint(0,2)] for j in range(F_LENGTH) ]
        # Get features as string
        features = ''.join(features)
        # Positive features contain pattern 'AAA' but not 'BBB'
        if 'AAA' in features and 'BBB' not in features:
            positive.append(features)
        # Negative features contain pattern 'BBB' but not 'AAA'
        elif 'BBB' in features and 'AAA' not in features:
            negative.append(features)

    # Positive features associates with positive label [1,0]
    positive = [ [f,[1,0]] for f in positive ]

    # Negative features associates with negative label [0,1]
    negative = [ [f,[0,1]] for f in negative ]

    # Concatenate posive and negative lists of samples
    dataset = positive + negative

    # Shuffle dataset before split
    random.shuffle(dataset)

    # Return dsets = [dtrain,dtest,dval]
    dsets = cio.dataset_to_dsets(dataset,runData,prefix='dummy_strings',encode=True)

    # dtrain, dtest, dval are dataSet EpyNN objects
    dtrain = dsets[0]

    # dtrain.X (features) is a numpy array of shape (N_SAMPLES//2,F_LENGTH,vocab_size)
    print (dtrain.X.shape) # (500, 11, 3)
    # dtrain.X[0] is a numpy array of shape (F_LENGTH,)
    print (dtrain.X[0].shape) # (11, 3)
    # dtrain.X[0] contains one encoded sample features
    print (dtrain.X[0]) # [ [0. 1. 0.] [0. 1. 0.] ... [0. 1. 0.] ]
    # Note that you can decode sample features
    print (cio.one_hot_decode_sequence(dtrain.X[0],runData)) # AAAACABACCA

    # dtrain.Y (label) is a numpy array of shape (N_SAMPLES//2,N_LABELS)
    print (dtrain.Y.shape) # (500, 2)
    # dtrain.Y[0] is a numpy array of shape (N_LABELS,)
    print (dtrain.X[0].shape) # (2,)
    # dtrain.Y[0] contains one sample label
    print (dtrain.Y[0])

    return dsets
