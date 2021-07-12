#EpyNN/nnlibs/commons/library.py
from nnlibs.commons.decorators import *

import numpy as np
import pickle
import random
import shutil
import glob
import os


@log_function
def read_pickle(f):
    with open(f, 'rb') as msg:
        c = pickle.load(msg)
    return c


@log_function
def read_file(f):
    with open(f, 'r') as msg:
        c = msg.read()
    return c


#@log_function
def write_pickle(f,c):
    with open(f, 'wb') as msg:
        pickle.dump(c,msg)


@log_function
def init_dir(CFG=None):

    if not os.path.exists('./datasets'):
        os.mkdir('./datasets')

    if not os.path.exists('./models'):
        os.mkdir('./models')

    if CFG:

        if CFG['directory_clear'] == True:

            shutil.rmtree('./datasets')
            os.mkdir('./datasets')

            shutil.rmtree('./models')
            os.mkdir('./models')


#@log_function
def check_and_write(model,hPars,runData):

    # Load metrics for target dataset (see ./metrics.py)
    metrics = runData.s[runData.m['m']][runData.m['d']]

    # Evaluate metrics
    if max(metrics) == metrics[-1] and runData.b['ms']:

        if runData.b['ms'] == True:

            data = {
                        'model': model,
                    }

            write_pickle(runData.p['ms'],data)

            runData.b['s'] = True

    return None


@log_function
def read_model(model_path=None):

    if model_path == None:
        model_path = max(glob.glob('./models/*'), key=os.path.getctime)

    model = read_pickle(model_path)['model']

    return model
