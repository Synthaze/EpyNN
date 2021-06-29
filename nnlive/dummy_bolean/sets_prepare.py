#EpyNN/nnlive/dummy_bolean/sets_prepare.py
import nnlibs.commons.io as cio

import random


def sets_prepare(runData):

    if runData.m['s'] == None:
        runData.m['s'] = 1000

    positive = [ [1,1,1,1,1,1,1,1,1,1] for x in range(runData.m['s']) ]
    negative = [ [0,0,0,0,0,0,0,0,0,0] for x in range(runData.m['s']) ]

    # Assign label (or class) to positive [1,0] and negative [0,1] samples
    positive = [ [x,[1,0]] for x in positive ]
    negative = [ [x,[0,1]] for x in negative ]

    # Merge positive and negative data
    dataset = positive + negative

    random.shuffle(dataset)

    dsets = cio.dataset_to_dsets(dataset,runData,prefix='dummy_bolean',encode=False)

    return dsets
