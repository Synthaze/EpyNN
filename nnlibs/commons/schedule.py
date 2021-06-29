#EpyNN/nnlibs/commons/schedule.py
import nnlibs.commons.logs as clo

import numpy as np


def exp_decay(hPars):

    return [ hPars.s['l'] * hPars.s['d']**(x//hPars.s['c']) * np.exp(-(x%hPars.s['c'])*hPars.s['k']) for x in range(hPars.i) ]


def lin_decay(hPars):

    return [ hPars.s['l'] / (1 + hPars.s['k'] * 100 * x) for x in range(hPars.i) ]


def steady(hPars):

    return [ hPars.s['l'] for x in range(hPars.i) ]


def schedule_mode(hPars):

    schedulers = {
    'steady': steady,
    'exp_decay': exp_decay,
    'lin_decay': lin_decay,
    }

    hPars.l = schedulers[hPars.s['s']](hPars)

    clo.log_lr_schedule(hPars)

    return hPars.l
