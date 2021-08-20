# EpyNN/nnlibs/commons/loss.py
# Related third party imports
import numpy as np


# To prevent from divide floatting points errors
E_SAFE = 1e-10


def loss_functions(key=None):
    """Callback function for loss.

    :param key: Name of the loss function, defaults to `None` which returns all functions.
    :type key: str, optional

    :return: Loss functions or computed loss.
    :rtype: dict[str, function] or :class:`numpy.ndarray`
    """
    loss = {
        'CCE': CCE,
        'BCE': BCE,
        'MSE': MSE,
        'MAE': MAE,
        'RMSLE': RMSLE,
    }
    # If key provided, returns output of function
    if key:
        loss = loss[key]

    return loss


def CCE(Y, A, deriv=False):
    """Categorical Cross-Entropy.

    :param Y: True labels for a set of samples.
    :type Y: :class:`numpy.ndarray`

    :param A: Output of forward propagation.
    :type A: :class:`numpy.ndarray`

    :param deriv: To compute the derivative.
    :type deriv: bool, optional

    :return: loss.
    :rtype: :class:`numpy.ndarray`
    """
    if not deriv:
        loss = -(Y * np.log(A + E_SAFE))

    elif deriv:
        loss = -(Y - A + E_SAFE)

    return loss


def BCE(Y, A, deriv=False):
    """Binary Cross-Entropy.

    :param Y: True labels for a set of samples.
    :type Y: :class:`numpy.ndarray`

    :param A: Output of forward propagation.
    :type A: :class:`numpy.ndarray`

    :param deriv: To compute the derivative.
    :type deriv: bool, optional

    :return: loss.
    :rtype: :class:`numpy.ndarray`
    """
    if not deriv:
        loss = -(Y*np.log(A+E_SAFE) + (1-Y)*np.log((1-A)+E_SAFE))

    elif deriv:
        loss = -(Y/(A+E_SAFE) - (1-Y)/((1-A)+E_SAFE))

    return loss


def MAE(Y, A, deriv=False):
    """Mean Absolute Error.

    :param Y: True labels for a set of samples.
    :type Y: :class:`numpy.ndarray`

    :param A: Output of forward propagation.
    :type A: :class:`numpy.ndarray`

    :param deriv: To compute the derivative.
    :type deriv: bool, optional

    :return: loss.
    :rtype: :class:`numpy.ndarray`
    """
    if not deriv:
        loss = np.abs(Y - A)

    elif deriv:
        loss = -((Y-A) / np.abs(Y-A))

    return loss


def MSE(Y, A, deriv=False):
    """Mean Square Error.

    :param Y: True labels for a set of samples.
    :type Y: :class:`numpy.ndarray`

    :param A: Output of forward propagation.
    :type A: :class:`numpy.ndarray`

    :param deriv: To compute the derivative.
    :type deriv: bool, optional

    :return: loss.
    :rtype: :class:`numpy.ndarray`
    """
    if not deriv:
        loss = np.square(Y - A + E_SAFE)

    elif deriv:
        loss = -2 * (Y-A)

    return loss


def RMSLE(Y, A, deriv=False):
    """Root Mean Square Logarythmic Error.

    :param Y: True labels for a set of samples.
    :type Y: :class:`numpy.ndarray`

    :param A: Output of forward propagation.
    :type A: :class:`numpy.ndarray`

    :param deriv: To compute the derivative.
    :type deriv: bool, optional

    :return: loss.
    :rtype: :class:`numpy.ndarray`
    """
    if not deriv:
        loss = np.sqrt(np.square(np.log1p(Y) - np.log1p(A)))

    elif deriv:
        loss = -(
                (np.log1p(Y) - np.log1p(A))
                / ((A+1) * np.sqrt(np.square(np.log1p(Y) - np.log1p(A))))
                )

    return loss
