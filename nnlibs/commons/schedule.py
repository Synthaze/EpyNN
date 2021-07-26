#EpyNN/nnlibs/commons/schedule.py
import nnlibs.commons.logs as clo

import numpy as np


def exp_decay(hPars):

    e, lr, n, k, epc = hPars

    return [ lr * d**(x//epc) * np.exp(-(x%epc)*k) for x in range(e) ]


def lin_decay(hPars):

    return [ lr / (1 + k * 100 * x) for x in range(e) ]


def steady(hPars):

    return [ lr for x in range(e) ]


def schedulers(se_hPars):

    e = se_hPars['training_epochs']
    lr = se_hPars['learning_rate']
    n = se_hPars['cycling_n']
    k = se_hPars['decay_k']

    epc = se_hPars['epochs_per_cycle'] = e // n

    # Default decay
    if k == 0:
        # 0.005% of initial lr for last epoch in cycle
        k = se_hPars['decay_k'] = 10 / epc


    schedulers = {
    'steady': steady,
    'exp_decay': exp_decay,
    'lin_decay': lin_decay,
    }

    hPars = (e, lr, n, k, epc)

    lrate = schedulers[se_hPars['schedule_mode']](hPars)

    return se_hPars, lrate
