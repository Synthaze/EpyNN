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
    """Read pickle binary file.

    :param f: Filename
    :type f: str

    :return: File content
    :rtype: str
    """
    with open(f, 'rb') as msg:
        c = pickle.load(msg)

    return c


def read_file(f):
    """Read text file.

    :param f: Filename
    :type f: str

    :return: File content
    :rtype: str
    """
    with open(f, 'r') as msg:
        c = msg.read()

    return c


def write_pickle(f, c):
    """Write pickle binary file.

    :param f: Filename
    :type f: str

    :param c: Content to write
    :type c: object
    """
    with open(f, 'wb') as msg:
        pickle.dump(c,msg)

    return None


def configure_directory(clear=False):
    """Configure working directory.

    :param se_config: Settings for general configuration
    :type se_config: dict
    """
    datasets_path = os.path.join(os.getcwd(), 'datasets')
    models_path = os.path.join(os.getcwd(), 'models')
    plots_path = os.path.join(os.getcwd(), 'plots')

    for path in [datasets_path, models_path, plots_path]:

        if clear and os.path.exists(path):
            shutil.rmtree(path)
            process_logs('Remove: '+path, level=2)

        if not os.path.exists(path):
            os.mkdir(path)
            process_logs('Make: '+path, level=1)

    return None


def write_model(model, model_path=None):
    """Write EpyNN model on disk.

    :param model: An instance of EpyNN network.
    :type model: :class:`nnlibs.meta.models.EpyNN`

    :param model_path: Location to write model
    :type model_path: str
    """
    data = {
                'model': model,
            }

    if model_path:
        pass
    else:
        model_path = os.path.join(os.getcwd(), 'models', model.uname)
        model_path = model_path+'.pickle'

    write_pickle(model_path, data)
    process_logs('Make: ' + model_path, level=1)

    return None


# def check_and_write(model,hPars,runData):
#     """.
#     """
#     # Load metrics for target dataset (see ./metrics.py)
#     metrics = runData.s[runData.m['m']][runData.m['d']]
#
#     # Evaluate metrics
#     if max(metrics) == metrics[-1] and metrics.count(metrics[-1]) == 1 and runData.b['ms']:
#
#         if runData.b['ms'] == True:
#
#             data = {
#                         'model': model,
#                     }
#
#             write_pickle(runData.p['ms'],data)
#
#             runData.b['s'] = True
#
#     return None


def read_model(model_path=None):
    """Read EpyNN model from disk.

    :param model_path: Model location.
    :type model_path: str
    """
    if model_path:
        pass
    else:
        models_path = os.path.join(os.getcwd(), 'models', '*')
        model_path = max(glob.glob(models_path), key=os.path.getctime)

    model = read_pickle(model_path)['model']

    return model


def read_dataset(dataset_path=None):
    """Read dataset from disk.

    :param dataset_path: Dataset location.
    :type dataset_path: str
    """
    if dataset_path:
        pass
    else:
        dataset_path = os.path.join(os.getcwd(), 'datasets', '*')
        dataset_path = max(glob.glob(dataset_path), key=os.path.getctime)

    dataset = read_pickle(dataset_path)

    return dataset


def settings_verification():
    """Import default settings if not present in working directory.
    """
    init_path = str(pathlib.Path(__file__).parent.parent.absolute())

    if not os.path.exists('settings.py'):
        se_default_path = os.path.join(init_path, 'settings.py')
        shutil.copy(se_default_path, 'settings.py')

    return None
