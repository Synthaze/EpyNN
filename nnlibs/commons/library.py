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

    if not os.path.exists('./sets'):
        os.mkdir('./sets')

    if not os.path.exists('./models'):
        os.mkdir('./models')

    if CFG:
        
        if CFG['directory_clear'] == True:

            shutil.rmtree('./models')
            os.mkdir('./models')

            shutil.rmtree('./sets')
            os.mkdir('./sets')

#@log_function
def check_and_write(model,dsets,hPars,runData):

    # Load metrics for target dataset (see ./metrics.py)
    metrics = runData.s[runData.m['m']][runData.m['d']]
    # Evaluate metrics
    if min(metrics) == metrics[-1] and runData.b['ms']:

        data = [model,dsets,hPars,runData]

        bool = ['model_save','dsets_save','hPars_save','runData_save']

        for i, _save in enumerate(bool):

            if runData.c[_save] == False:

                data[i] = None

        data = {
                    'model': data[0],
                    'dsets': data[1],
                    'hPars': data[2],
                    'runData': data[3],
                }

        write_pickle(runData.p['ms'],data)

        runData.b['s'] = True

    return None


def read_data(path=None):

    if path == None:

        path = max(glob.glob('./models/*'), key=os.path.getctime)

    data = read_pickle(path)

    return data
