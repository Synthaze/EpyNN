# EpyNN/nnlibs/commons/schedule.py
# Related third party imports
import numpy as np


def schedulers(se_hPars, training_epochs):
    """Learning rate schedule.

    :param se_hPars: Hyperparameters settings for layer
    :type se_hPars: dict

    :param training_epochs: Number of training epochs for model
    :type training_epochs: int

    :return: Updated settings for layer hyperparameters
    :rtype: dict

    :return: Scheduled learning rate for layer
    :rtype: list
    """
    e = se_hPars['training_epochs'] = training_epochs
    lr = se_hPars['learning_rate']
    n = se_hPars['cycling_n']
    d = se_hPars['descent_d']
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

    hPars = (e, lr, n, k, d, epc)

    lrate = schedulers[se_hPars['schedule_mode']](hPars)

    return se_hPars, lrate


def exp_decay(hPars):
    """Exponential decay schedule for learning rate.

    :param hPars: Contains hyperparameters.
    :type hPars: tuple

    :return: Scheduled learning rate
    :rtype: list
    """
    e, lr, n, k, d, epc = hPars

    lrate = [lr * d**(x//epc) * np.exp(-(x%epc) * k) for x in range(e)]

    return lrate


def lin_decay(hPars):
    """Linear decay schedule for learning rate.

    :param hPars: Contains hyperparameters.
    :type hPars: tuple

    :return: Scheduled learning rate
    :rtype: list
    """
    e, lr, n, k, d, epc = hPars

    lrate = [lr / (1 + k*100*x) for x in range(e)]

    return lrate


def steady(hPars):
    """Steady schedule for learning rate.

    :param hPars: Contains hyperparameters.
    :type hPars: tuple

    :return: Scheduled learning rate
    :rtype: list
    """
    e, lr, n, k, d, epc = hPars

    lrate = [lr for x in range(e)]

    return lrate
