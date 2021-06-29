#EpyNN/nnlibs/commons/library.py
from nnlibs.commons.decorators import *

import pickle
import random
import shutil
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
def read_set_fasta(f):
    with open(f,'r') as fp:
        d = list(set([ x for x in fp.read().splitlines() if 'B' not in x and 'O' not in x and 'Z' not in x and 'X' not in x and 'U' not in x]))
    random.shuffle(d)
    return d


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


#@log_function
def check_and_write(model,dsets,hPars,runData):

    metrics = runData.s[runData.m['m']][runData.m['d']]

    if min(metrics) == metrics[-1] and runData.b['ms']:

        data = {
                    'model': model,
                    'dsets': dsets,
                    'hPars': hPars,
                    'runData': runData,
                }

        write_pickle(runData.p['ms'],data)

        runData.b['s'] = True

    return None
