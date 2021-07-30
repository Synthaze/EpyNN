# EpyNN/nnlibs/commons/maths.py
# Standard library imports
import random

# Related third party imports
import numpy as np


### Activation functions and derivatives

# Rectifier Linear Unit (ReLU)

def relu(x, deriv=False):
    """Compute ReLU activation or derivative

    :param x:
    :type x: class:`numpy.ndarray`

    :param deriv:
    :type deriv: bool

    :return:
    :rtype: class:`numpy.ndarray`
    """

    if deriv == False:
        x = np.maximum(0, x)

    elif deriv == True:
        x = np.greater(x, 0).astype(int)

    return x


# Leaky Rectifier Linear Unit (LReLU)

def lrelu(x, deriv=False):
    """Compute LReLU activation or derivative

    :param x:
    :type x: class:`numpy.ndarray`

    :param deriv:
    :type deriv: bool

    :return:
    :rtype: class:`numpy.ndarray`
    """

    a = CST['l']

    if deriv == False:
        x = np.maximum(a * x, x)

    elif deriv == True:
        x = np.where(x > 0, 1, a)

    return x


# Exponential Linear Unit (ELU)

def elu(x, deriv=False):
    """Compute ELU activation or derivative

    :param x:
    :type x: class:`numpy.ndarray`

    :param deriv:
    :type deriv: bool

    :return:
    :rtype: class:`numpy.ndarray`
    """

    a = CST['e']

    if deriv == False:
        x = np.where(x > 0, x, a * (np.exp(x, where=x<=0)-1))

    elif deriv == True:
        x = np.where(x > 0, 1, elu(x) + a)

    return x


# Swish

def swish(x, deriv=False):
    """Compute Swish activation or derivative

    :param x:
    :type x: class:`numpy.ndarray`

    :param deriv:
    :type deriv: bool

    :return:
    :rtype: class:`numpy.ndarray`
    """

    if deriv == False:
        x = x / (1-np.exp(-x))

    elif deriv == True:
        x = None

    return x


# Sigmoid (σ)

def sigmoid(x, deriv=False):
    """Compute Sigmoid activation or derivative

    :param x:
    :type x: class:`numpy.ndarray`

    :param deriv:
    :type deriv: bool

    :return:
    :rtype: class:`numpy.ndarray`
    """

    if deriv == False:
        x = np.where(
                    x >= 0, # condition
                    1 / (1+np.exp(-x)), # For positive values
                    np.exp(x) / (1+np.exp(x)) # For negative values
                    )

    elif deriv == True:
        x = sigmoid(x)
        x = x * (1-x)

    return x


# Hyperbolic tangent (tanh)

def tanh(x, deriv=False):
    """Compute tanh activation or derivative

    :param x:
    :type x: class:`numpy.ndarray`

    :param deriv:
    :type deriv: bool

    :return:
    :rtype: class:`numpy.ndarray`
    """

    x = (2 / (1+np.exp(-2*x))) - 1

    if deriv == False:
        pass

    elif deriv == True:
        x = 1 - x**2

    return x


# Softmax

def softmax(x, deriv=False):
    """Compute softmax activation or derivative

    :param x:
    :type x: class:`numpy.ndarray`

    :param deriv:
    :type deriv: bool

    :return:
    :rtype: class:`numpy.ndarray`
    """

    #T = CST['s']
    T = 1

    if deriv == False:

        x = x - np.max(x, axis=0, keepdims=True)

        x = np.exp(x / T)
        x_ = np.sum(x, axis=0, keepdims=True)
        x_ = x / x_

    elif deriv == True:

        pass

    return x


# Weights initialization
def xavier(shape, rng=np.random):
    """.
    """
    x = rng.standard_normal(shape)
    x /= np.sqrt(shape[1])
    return x


def orthogonal(shape, rng=np.random):
    """.
    """
    p = rng.standard_normal(shape)

    if shape[0] < shape[1]:
        p = p.T
    # Compute QR factorization
    q, r = np.linalg.qr(p)
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = np.diag(r, 0)
    ph = np.sign(d)
    q *= ph

    if shape[0] < shape[1]:
        q = q.T

    p = q

    return p


def clip_gradient(layer,max_norm=0.25):
    """.
    """
    # Set the maximum of the norm to be of type float
    max_norm = float(max_norm)
    total_norm = 0

    # Calculate the L2 norm squared for each gradient and add them to the total norm
    for grad in layer.g.values():
        grad_norm = np.sum(np.power(grad, 2))
        total_norm += grad_norm

    total_norm = np.sqrt(total_norm)

    # Calculate clipping coeficient
    clip_coef = max_norm / (total_norm + 1e-6)

    # If the total norm is larger than the maximum allowable norm, then clip the gradient
    if clip_coef < 1:
        for g in layer.g.keys():
            layer.g[g] *= clip_coef

    return None
