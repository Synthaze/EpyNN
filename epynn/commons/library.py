# EpyNN/epynn/commons/library.py
# Standard library imports
import pathlib
import pickle
import shutil
import glob
import os
 
# Related third party imports
import numpy as np

# Local application/library specific imports
from epynn.commons.logs import process_logs


def read_pickle(f):
    """Read pickle binary file.

    :param f: Filename.
    :type f: str

    :return: File content.
    :rtype: Object
    """
    with open(f, 'rb') as msg:
        c = pickle.load(msg)

    return c


def read_file(f):
    """Read text file.

    :param f: Filename.
    :type f: str

    :return: File content.
    :rtype: str
    """
    with open(f, 'r') as msg:
        c = msg.read()

    return c


def write_pickle(f, c):
    """Write pickle binary file.

    :param f: Filename.
    :type f: str

    :param c: Content to write.
    :type c: Object
    """
    with open(f, 'wb') as msg:
        pickle.dump(c,msg)

    return None


def configure_directory(clear=False):
    """Configure working directory.

    :param clear: Remove and make directories, defaults to False.
    :type clear: bool, optional
    """
    # Set paths for defaults directories
    datasets_path = os.path.join(os.getcwd(), 'datasets')
    models_path = os.path.join(os.getcwd(), 'models')
    plots_path = os.path.join(os.getcwd(), 'plots')

    # Iterate over directory paths
    for path in [datasets_path, models_path, plots_path]:

        # If clear set to True, remove directories
        if clear and os.path.exists(path):
            shutil.rmtree(path)
            process_logs('Remove: '+path, level=2)

        # Create directory if not existing
        if not os.path.exists(path):
            os.mkdir(path)
            process_logs('Make: '+path, level=1)

    return None


def write_model(model, model_path=None):
    """Write EpyNN model on disk.

    :param model: An instance of EpyNN network object.
    :type model: :class:`epynn.network.models.EpyNN`

    :param model_path: Where to write model, defaults to `None` which sets path in `models` directory.
    :type model_path: str or NoneType, optional
    """
    data = {
                'model': model,
            }

    if model_path:
        # If model_path not set to None, pass on user-defined path
        pass
    else:
        # Set default location and name to write model on disk
        model_path = os.path.join(os.getcwd(), 'models', model.uname)
        model_path = model_path+'.pickle'

    # Write model with pickle
    write_pickle(model_path, data)
    process_logs('Make: ' + model_path, level=1)

    return None


def read_model(model_path=None):
    """Read EpyNN model from disk.

    :param model_path: Where to read model from, defaults to `None` which reads the last saved model in `models` directory.
    :type model_path: str or NoneType, optional
    """
    if model_path:
        # If model_path not set to None, pass on user-defined path
        pass
    else:
        # Set default location and name to read the model from
        models_path = os.path.join(os.getcwd(), 'models', '*')
        model_path = max(glob.glob(models_path), key=os.path.getctime)

    model = read_pickle(model_path)['model']

    return model


def settings_verification():
    """Import default :class:`epynn.settings.se_hPars` if not present in working directory.
    """
    # Absolute path of epynn directory
    init_path = str(pathlib.Path(__file__).parent.parent.absolute())

    # Copy defaults settings in working directory if not present
    if not os.path.exists('settings.py'):
        se_default_path = os.path.join(init_path, 'settings.py')
        shutil.copy(se_default_path, 'settings.py')

    return None
