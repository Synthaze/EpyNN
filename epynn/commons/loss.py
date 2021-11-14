# EpyNN/epynn/commons/loss.py
# Related third party imports
import numpy as np


# To prevent from divide floatting points errors
E_SAFE = 1e-16
 

def loss_functions(key=None, output_activation=None):
    """Callback function for loss.

    :param key: Name of the loss function, defaults to `None` which returns all functions.
    :type key: str, optional

    :param output_activation: Name of the activation function for output layer.
    :type output_activation: str, optional

    :raises Exception: If key is `CCE` and output activation is different from softmax.

    :raises Exception: If key is either `CCE`, `BCE` or `MSLE` and output activation is tanh.

    :return: Loss functions or computed loss.
    :rtype: dict[str, function] or :class:`numpy.ndarray`
    """
    loss = {
        'MSE': MSE,
        'MAE': MAE,
        'MSLE': MSLE,
        'CCE': CCE,
        'BCE': BCE,
    }
    # If key provided, returns output of function
    if key:
        loss = loss[key]

    if key == 'CCE' and output_activation != 'softmax':
        raise Exception('CCE can not be used with %s activation, \
                        please use softmax instead.' % output_activation)

    if key in ['CCE', 'BCE', 'MSLE'] and output_activation == 'tanh':
        raise Exception('%s contains log() not be used with %s activation, \
                         please change.' % (key, output_activation))

    return loss


def MSE(Y, A, deriv=False):
    """Mean Squared Error.

    :param Y: True labels for a set of samples.
    :type Y: :class:`numpy.ndarray`

    :param A: Output of forward propagation.
    :type A: :class:`numpy.ndarray`

    :param deriv: To compute the derivative.
    :type deriv: bool, optional

    :return: Loss.
    :rtype: :class:`numpy.ndarray`
    """
    U = A.shape[1]    # Number of output nodes

    if not deriv:
        loss =  1. / U * np.sum((Y - A)**2, axis=1)

    elif deriv:
        loss = -2. / U * (Y-A)

    return loss


def MAE(Y, A, deriv=False):
    """Mean Absolute Error.

    :param Y: True labels for a set of samples.
    :type Y: :class:`numpy.ndarray`

    :param A: Output of forward propagation.
    :type A: :class:`numpy.ndarray`

    :param deriv: To compute the derivative.
    :type deriv: bool, optional

    :return: Loss.
    :rtype: :class:`numpy.ndarray`
    """
    U = A.shape[1]    # Number of output nodes

    if not deriv:
        loss =  1. / U * np.sum(np.abs(Y-A), axis=1)

    elif deriv:
        loss = -1. / U * (Y-A) / (np.abs(Y-A)+E_SAFE)

    return loss


def MSLE(Y, A, deriv=False):
    """Mean Squared Logarythmic Error.

    :param Y: True labels for a set of samples.
    :type Y: :class:`numpy.ndarray`

    :param A: Output of forward propagation.
    :type A: :class:`numpy.ndarray`

    :param deriv: To compute the derivative.
    :type deriv: bool, optional

    :return: Loss.
    :rtype: :class:`numpy.ndarray`
    """
    U = A.shape[1]    # Number of output nodes

    if not deriv:
        loss = 1. / U * np.sum(np.square(np.log1p(Y) - np.log1p(A)), axis=1)

    elif deriv:
        loss = -2. / U * (np.log1p(Y) - np.log1p(A)) / (A + 1.)

    return loss


def CCE(Y, A, deriv=False):
    """Categorical Cross-Entropy.

    :param Y: True labels for a set of samples.
    :type Y: :class:`numpy.ndarray`

    :param A: Output of forward propagation.
    :type A: :class:`numpy.ndarray`

    :param deriv: To compute the derivative.
    :type deriv: bool, optional

    :return: Loss.
    :rtype: :class:`numpy.ndarray`
    """
    U = A.shape[1]    # Number of output nodes

    if not deriv:
        loss = -1. * np.sum(Y * np.log(A+E_SAFE), axis=1)

    elif deriv:
        loss = -1. * (Y / A)

    return loss


def BCE(Y, A, deriv=False):
    """Binary Cross-Entropy.

    :param Y: True labels for a set of samples.
    :type Y: :class:`numpy.ndarray`

    :param A: Output of forward propagation.
    :type A: :class:`numpy.ndarray`

    :param deriv: To compute the derivative.
    :type deriv: bool, optional

    :return: Loss.
    :rtype: :class:`numpy.ndarray`
    """
    U = A.shape[1]    # Number of output nodes

    if not deriv:
        loss = -1. / U * np.sum(Y*np.log(A+E_SAFE) + (1-Y)*np.log((1-A)+E_SAFE), axis=1)

    elif deriv:
        loss = 1. / U * (A-Y) / (A - A*A + E_SAFE)

    return loss
