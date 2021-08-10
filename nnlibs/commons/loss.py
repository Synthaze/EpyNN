# EpyNN/nnlibs/commons/loss.py
# Related third party imports
import numpy as np


#
E_SAFE = 1e-10


def loss_functions(key=None):

    loss = {
        'CCE': CCE,
        'BCE': BCE,
        'MSE': MSE,
        'MAE': MAE,
        'RMSLE': RMSLE,
    }

    if key:
        loss = loss[key]

    return loss


def CCE(Y, A, deriv=False):
    """Categorical Cross-Entropy.

    """
    if not deriv:
        loss = -(Y * np.log(A + E_SAFE))

    elif deriv:
        loss = -(Y - A + E_SAFE)

    return loss


def BCE(Y, A, deriv=False):
    """Binary Cross-Entropy.

    """
    if not deriv:
        loss = -(Y*np.log(A+E_SAFE) + (1-Y)*np.log((1-A)+E_SAFE))

    elif deriv:
        loss = -(Y/(A+E_SAFE) - (1-Y)/((1-A)+E_SAFE))

    return loss


def MAE(Y, A, deriv=False):
    """Mean Absolute Error.
    """
    if not deriv:
        loss = np.abs(Y - A)

    elif deriv:
        loss = -((Y-A) / np.abs(Y-A))

    return loss


def MSE(Y, A, deriv=False):
    """Mean Square Error.
    """
    if not deriv:
        loss = np.square(Y - A + E_SAFE)

    elif deriv:
        loss = -2 * (Y-A)

    return loss


def RMSLE(Y, A, deriv=False):
    """Root Mean Square Logarythmic Error.
    """
    if not deriv:
        loss = np.sqrt(np.square(np.log1p(Y) - np.log1p(A)))

    elif deriv:
        loss = -(
                (np.log1p(Y) - np.log1p(A))
                / ((A+1) * np.sqrt(np.square(np.log1p(Y) - np.log1p(A))))
                )

    return loss
