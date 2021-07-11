#EpyNN/nnlive/dummy_boolean/prepare_dataset.py
import nnlibs.commons.library as cli

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

    # Number of features describing a sample
    N_FEATURES = 11

    # Iterate over N_SAMPLES
    for i in range(N_SAMPLES):
        # Random choice True or False for N_FEATURES iterations
        features = [ random.choice([True,False]) for j in range(N_FEATURES) ]

        # Test if features associates with p_label (+)
        if features.count(True) > features.count(False):
            sample = [features,p_label]

        # Test if features associates with n_label (-)
        elif features.count(True) < features.count(False):
            sample = [features,n_label]

        # Append sample to dataset
        dataset.append(sample)

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


def show_data_shapes(dsets):

    # Local pretty print function
    def pprint(k,v,end='\n'):
        print ('{:<20}{:<}'.format(k,str(v)),end=end)

    # dsets in a list of nnlibs.commons.models.dataSet objects
    for dset in dsets:

        pprint('dataSet.n',dset.n) # Name of dset

        pprint('dataSet.id.shape',dset.id.shape) # ids array shape
        pprint('dataSet.id[0]',dset.id[0]) # sample id at index 0

        pprint('dataSet.X.shape',dset.X.shape) # X (features) array shape
        pprint('dataSet.X[0].shape',dset.X[0].shape) # X shape for sample 0
        pprint('dataSet.X[0]',dset.X[0]) # X inputs for sample 0
        pprint('dataSet.X[0]*1',dset.X[0]*1) # Boolean arithmetics

        pprint('dataSet.Y.shape',dset.Y.shape) # Y (label) array shape
        pprint('dataSet.Y[0].shape',dset.Y[0].shape) # Y shape for sample 0
        pprint('dataSet.Y[0]',dset.Y[0]) # Y one-hot encoded label sample 0

        pprint('dataSet.y.shape',dset.y.shape) # y (decoded label) array shape
        pprint('dataSet.y[0]',dset.y[0],end='\n\n') # y integer label for sample 0

    return None
