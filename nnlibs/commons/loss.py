# EpyNN/nnlibs/commons/loss.py
# Related third party imports
import numpy as np


# To prevent from divide floatting points errors
E_SAFE = 1e-16


class Loss:
    """.
    """

    def __init__(self, loss, output_layer):

        self.iloss = loss
        self.loss = loss_functions(key=loss)

        output_activation = output_layer.activation['activate']
        self.softmax = True if output_activation == 'softmax' else False

        return None

    def call(self, Y, A, deriv=False):

        N = A.shape[1]

        losses = self.loss(Y, A, deriv=deriv)

        if deriv and self.softmax:
            if self.iloss != 'CCE':
                losses *= (1 + 1/(N-1))

        return losses


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
        'MSLE': MSLE,
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
    N = A.shape[1]

    if not deriv:
        loss = -1. * np.sum(Y * np.log(A+E_SAFE), axis=1)

    elif deriv:
        loss = 1. * (A-Y) / (A - A*A + E_SAFE)

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
    N = A.shape[1]

    if not deriv:
        loss = -1. / N * np.sum(Y*np.log(A+E_SAFE) + (1-Y)*np.log((1-A)+E_SAFE), axis=1)

    elif deriv:
        loss = 1. / N * (A-Y) / (A - A*A + E_SAFE)

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
    N = A.shape[1]

    if not deriv:
        loss =  1. / N * np.sum(np.abs(Y-A), axis=1)

    elif deriv:
        loss = -1. / N * (Y-A) / (np.abs(Y-A)+E_SAFE)

    return loss


def MSE(Y, A, deriv=False):
    """Mean Squared Error.

    :param Y: True labels for a set of samples.
    :type Y: :class:`numpy.ndarray`

    :param A: Output of forward propagation.
    :type A: :class:`numpy.ndarray`

    :param deriv: To compute the derivative.
    :type deriv: bool, optional

    :return: loss.
    :rtype: :class:`numpy.ndarray`
    """
    N = A.shape[1]

    if not deriv:
        loss =  1. / N * np.sum((Y - A)**2, axis=1)

    elif deriv:
        loss = -2. / N * (Y-A)

    return loss


def MSLE(Y, A, deriv=False):
    """Mean Squared Logarythmic Error.

    :param Y: True labels for a set of samples.
    :type Y: :class:`numpy.ndarray`

    :param A: Output of forward propagation.
    :type A: :class:`numpy.ndarray`

    :param deriv: To compute the derivative.
    :type deriv: bool, optional

    :return: loss.
    :rtype: :class:`numpy.ndarray`
    """
    N = A.shape[1]

    if not deriv:
        loss = 1. / N * np.sum(np.square(np.log1p(Y) - np.log1p(A)), axis=1)

    elif deriv:
        loss = -2. / N * (np.log1p(Y) - np.log1p(A)) / (A + 1.)

    return loss
