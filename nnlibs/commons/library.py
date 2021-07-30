# EpyNN/nnlibs/commons/library.py
# Standard library imports
import pathlib
import pickle
import random
import shutil
import glob
import os

# Related third party imports
import numpy as np

# Local application/library specific imports
from nnlibs.commons.logs import process_logs


def read_pickle(f):
    """.
    """
    with open(f, 'rb') as msg:
        c = pickle.load(msg)

    return c


def read_file(f):
    """.
    """
    with open(f, 'r') as msg:
        c = msg.read()

    return c


def write_pickle(f, c):
    """.
    """
    with open(f, 'wb') as msg:
        pickle.dump(c,msg)

    return None


def configure_directory(se_config=None):
    """.
    """
    datasets_path = os.path.join(os.getcwd(), 'datasets')
    models_path = os.path.join(os.getcwd(), 'models')

    if se_config:

        if se_config['directory_clear'] == True:

            shutil.rmtree(datasets_path)
            process_logs('Remove: '+datasets_path, level=2)

            shutil.rmtree(models_path)
            process_logs('Remove: '+models_path, level=2)

    if not os.path.exists('datasets'):
        os.mkdir(datasets_path)
        process_logs('Make: '+datasets_path, level=1)

    if not os.path.exists('models'):
        os.mkdir(models_path)
        process_logs('Make: '+models_path, level=1)

    return None


def write_model(model):

    models_path = os.path.join(os.getcwd(), 'models', model.uname)

    data = {
                'model': model,
            }

    path = models_path+'.pickle'

    write_pickle(path, data)
    process_logs('Make: '+path, level=1)

    return None


def check_and_write(model,hPars,runData):
    """.
    """
    # Load metrics for target dataset (see ./metrics.py)
    metrics = runData.s[runData.m['m']][runData.m['d']]

    # Evaluate metrics
    if max(metrics) == metrics[-1] and metrics.count(metrics[-1]) == 1 and runData.b['ms']:

        if runData.b['ms'] == True:

            data = {
                        'model': model,
                    }

            write_pickle(runData.p['ms'],data)

            runData.b['s'] = True

    return None


def read_model(model_path=None):
    """.
    """
    models_path = os.path.join(os.getcwd(), 'models', '*')
    if model_path == None:
        model_path = max(glob.glob(models_path), key=os.path.getctime)

    model = read_pickle(model_path)['model']

    return model


def read_dataset(dataset_path=None):
    """.
    """
    # Get path most recent dataset
    if dataset_path == None:
        dataset_path = max(glob.glob('./datasets/*'), key=os.path.getctime)

    # Read dataset
    dataset = read_pickle(dataset_path)

    return dataset


def settings_verification():

    init_path = str(pathlib.Path(__file__).parent.absolute())

    if not os.path.exists('settings.py'):
        se_default_path = os.path.join(init_path, 'settings.py')
        shutil.copy(se_default_path,'settings.py')

    return None
