#EpyNN/nnlive/template/sets_prepare.py
import nnlibs.commons.library as cl
import nnlibs.commons.io as cio

import random


def sets_prepare(runData):

    # Positive data are Homo sapiens O-GlcNAcylated peptide sequences from oglcnac.mcw.edu
    path_positive = 'path/to/positive.dat'
    # Negative data are peptide sequences presumably not O-GlcNAcylated
    path_negative = 'path/to/negative.dat'

    # Read text files, each containing one sequence per line
    positive = cl.read_file(path_positive).splitlines()
    negative = cl.read_file(path_negative).splitlines()

    # Shuffle data to prevent from any sorting previously applied
    random.shuffle(positive)
    random.shuffle(negative)

    # Default N_SAMPLES = runData.m['s'] is the length of positive data
    if runData.m['s'] == None:
        runData.m['s'] = len(positive)

    # Truncate positive to runData.m['s']
    positive = positive[:runData.m['s']]
    # Here we build a "balanced" dataset: with same number of positive and negative samples
    # Truncate negative to length of positive
    negative = negative[:len(positive)]

    # Assign label (or class) to positive [1,0] and negative [0,1] samples
    positive = [ [x,[1,0]] for x in positive ]
    negative = [ [x,[0,1]] for x in negative ]

    # Merge positive and negative data
    dataset = positive + negative

    random.shuffle(dataset)

    dsets = cio.dataset_to_dsets(dataset,runData,prefix='oglcnac',encode=True)

    return dsets
