#EpyNN/nnlive/dummy_bolean/sets_prepare.py
import nnlibs.commons.io as cio

import random


def sets_prepare(runData):
    """
    Prepare dummy dataset with bolean sample features

    sample = [features,label]

    features is a list of sample features of shape (F_LENGTH,)
    label is a one-hot encoded label of shape (N_LABELS,)
    """

    dataset = []

    # See settings.py - config dict is cached into runData.m
    N_SAMPLES = runData.m['s']

    # LENGTH is the number of features describing a sample
    F_LENGTH = 11

    # Iterate through N_SAMPLES
    for i in range(N_SAMPLES):
        # features is a 1D list of length F_LENGTH containing random boleans
        features = [ [True,False][random.randint(0,1)] for j in range(F_LENGTH) ]
        # Append X to dataset list object
        dataset.append(features)

    # Positive features contains more True
    positive = [ f for f in dataset if f.count(True) >  f.count(False) ]
    # Positive features associates with positive label [1,0]
    positive = [ [f,[1,0]] for f in positive ]

    # Negative features contains more False
    negative = [ f for f in dataset if f.count(True) <  f.count(False) ]
    # Negative features associates with negative label [0,1]
    negative = [ [f,[0,1]] for f in negative ]

    # Concatenate posive and negative lists of samples
    dataset = positive + negative

    # Shuffle dataset before split
    random.shuffle(dataset)

    # Return dsets = [dtrain,dtest,dval]
    dsets = cio.dataset_to_dsets(dataset,runData,prefix='dummy_bolean')

    # dtrain, dtest, dval are dataSet EpyNN objects
    dtrain = dsets[0]

    # dtrain.X (features) is a numpy array of shape (N_SAMPLES//2,F_LENGTH)
    print (dtrain.X.shape) # (500, 11)
    # dtrain.X[0] is a numpy array of shape (F_LENGTH,)
    print (dtrain.X[0].shape) # (11,)
    # dtrain.X[0] contains one sample features
    print (dtrain.X[0]) # [False False False ... False True]
    # Note that True evaluates to 1 and False to 0
    print (dtrain.X[0]*1) # [0 0 0 ... 0 1]

    # dtrain.Y (label) is a numpy array of shape (N_SAMPLES//2,N_LABELS)
    print (dtrain.Y.shape) # (500, 2)
    # dtrain.Y[0] is a numpy array of shape (N_LABELS,)
    print (dtrain.X[0].shape) # (2,)
    # dtrain.Y[0] contains one sample label
    print (dtrain.Y[0])

    return dsets
