# EpyNN/epynn/commons/schedule.py
# Related third party imports
import numpy as np
 

def schedule_functions(schedule, hPars):
    """Roots hyperparameters to relevant scheduler.

    :param schedule: Schedule mode.
    :type schedule: str

    :param hPars: Contains hyperparameters.
    :type hPars: tuple[int or float]

    :return: Scheduled learning rate.
    :rtype: list[float]
    """
    schedulers = {
        'exp_decay': exp_decay,
        'lin_decay': lin_decay,
        'steady': steady,
    }

    lrate = schedulers[schedule](hPars)

    return lrate


def exp_decay(hPars):
    """Exponential decay schedule for learning rate.

    :param hPars: Contains hyperparameters.
    :type hPars: tuple[int or float]

    :return: Scheduled learning rate.
    :rtype: list[float]
    """
    e, lr, n, k, d, epc = hPars

    lrate = [lr * (1-d) ** (x//epc) * np.exp(-(x%epc) * k) for x in range(e)]

    return lrate


def lin_decay(hPars):
    """Linear decay schedule for learning rate.

    :param hPars: Contains hyperparameters.
    :type hPars: tuple[int or float]

    :return: Scheduled learning rate.
    :rtype: list[float]
    """
    e, lr, n, k, d, epc = hPars

    lrate = [lr / (1 + k*100*x) for x in range(e)]

    return lrate


def steady(hPars):
    """Steady schedule for learning rate.

    :param hPars: Contains hyperparameters.
    :type hPars: tuple[int or float]

    :return: Scheduled learning rate.
    :rtype: list[float]
    """
    e, lr, n, k, d, epc = hPars

    lrate = [lr for x in range(e)]

    return lrate
