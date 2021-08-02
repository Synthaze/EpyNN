# EpyNN/nnlibs/commons/maths.py
# Standard library imports
import random

# Related third party imports
import numpy as np


E_SAFE = 1e-10


def activation_tune(se_hPars):
    """.
    """
    global layer_hPars
    layer_hPars = se_hPars


### Activation functions and derivatives

# Identity function

def identity(x, deriv=False):
    """Compute ReLU activation or derivative

    :param x: Input array to pass in function
    :type x: class:`numpy.ndarray`

    :param deriv: To compute derivative
    :type deriv: bool

    :return: Output array passed in function
    :rtype: class:`numpy.ndarray`
    """
    if not deriv:
        pass

    else:
        x = np.zeros_like(x)
    return x


# Rectifier Linear Unit (ReLU)

def relu(x, deriv=False):
    """Compute ReLU activation or derivative

    :param x: Input array to pass in function
    :type x: class:`numpy.ndarray`

    :param deriv: To compute derivative
    :type deriv: bool

    :return: Output array passed in function
    :rtype: class:`numpy.ndarray`
    """

    if not deriv:
        x = np.maximum(0, x)

    elif deriv:
        x = np.greater(x, 0).astype(int)

    return x


# Leaky Rectifier Linear Unit (LReLU)

def lrelu(x, deriv=False):
    """Compute LReLU activation or derivative

    :param x: Input array to pass in function
    :type x: class:`numpy.ndarray`

    :param deriv: To compute derivative
    :type deriv: bool

    :return: Output array passed in function
    :rtype: class:`numpy.ndarray`
    """

    a = layer_hPars['LRELU_alpha']

    if not deriv:
        x = np.maximum(a * x, x)

    elif deriv:
        x = np.where(x > 0, 1, a)

    return x


# Exponential Linear Unit (ELU)

def elu(x, deriv=False):
    """Compute ELU activation or derivative

    :param x: Input array to pass in function
    :type x: class:`numpy.ndarray`

    :param deriv: To compute derivative
    :type deriv: bool

    :return: Output array passed in function
    :rtype: class:`numpy.ndarray`
    """

    a = layer_hPars['ELU_alpha']

    if not deriv:
        x = np.where(x > 0, x, a * (np.exp(x, where=x<=0)-1))

    elif deriv:
        x = np.where(x > 0, 1, elu(x) + a)

    return x


# Sigmoid (σ)

def sigmoid(x, deriv=False):
    """Compute Sigmoid activation or derivative

    :param x: Input array to pass in function
    :type x: class:`numpy.ndarray`

    :param deriv: To compute derivative
    :type deriv: bool

    :return: Output array passed in function
    :rtype: class:`numpy.ndarray`
    """
    x = np.clip(x, -500, 500)
    if not deriv:

        x = np.where(
                    x >= 0, # condition
                    1 / (1+np.exp(-x)), # For positive values
                    np.exp(x) / (1+np.exp(x)) # For negative values
                    )

    elif deriv:
        x = sigmoid(x)
        x = x * (1-x)

    return x


# Swish

def swish(x, deriv=False):
    """Compute Swish activation or derivative

    :param x: Input array to pass in function
    :type x: class:`numpy.ndarray`

    :param deriv: To compute derivative
    :type deriv: bool

    :return: Output array passed in function
    :rtype: class:`numpy.ndarray`
    """

    if not deriv:
        x = x * sigmoid(x)

    elif deriv:
        pass

    return x


# Hyperbolic tangent (tanh)

def tanh(x, deriv=False):
    """Compute tanh activation or derivative

    :param x: Input array to pass in function
    :type x: class:`numpy.ndarray`

    :param deriv: To compute derivative
    :type deriv: bool

    :return: Output array passed in function
    :rtype: class:`numpy.ndarray`
    """

    if not deriv:
        x = (np.exp(2 * x)-1) / (np.exp(2 * x)+1)

    elif deriv:
        x = 1 - x**2

    return x


# Softmax

def softmax(x, deriv=False):
    """Compute softmax activation or derivative

    :param x: Input array to pass in function
    :type x: class:`numpy.ndarray`

    :param deriv: To compute derivative
    :type deriv: bool

    :return: Output array passed in function
    :rtype: class:`numpy.ndarray`
    """

    T = layer_hPars['softmax_temperature']

    if not deriv:
        x_safe = x - np.max(x, axis=1, keepdims=True)

        x_exp = np.exp(x_safe / T)
        x_sum = np.sum(x_exp, axis=1, keepdims=True)

        x = x_exp / x_sum

    elif deriv:
        x = softmax(x)
        x = x * (1-x)

    return x


### Weight initialization

def xavier(shape, rng=np.random):
    """Xavier initialization for weight array.

    :param shape: Shape of weight array
    :type shape: tuple

    :param rng: Pseudo-random number generator
    :type rng: :class:`numpy.random`

    :return: Initialized weight array
    :rtype: class:`numpy.ndarray`
    """
    W = rng.standard_normal(shape)
    W /= np.sqrt(shape[1])

    return W


def orthogonal(shape, rng=np.random):
    """Orthogonal initialization for weight array.

    :param shape: Shape of weight array
    :type shape: tuple

    :param rng: Pseudo-random number generator
    :type rng: :class:`numpy.random`

    :return: Initialized weight array
    :rtype: class:`numpy.ndarray`
    """
    W = rng.standard_normal(shape)

    W = W.T if shape[0] < shape[1] else W

    # Compute QR factorization
    q, r = np.linalg.qr(W)

    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = np.diag(r, 0)
    ph = np.sign(d)
    q *= ph

    W = q.T if shape[0] < shape[1] else q

    return W


def clip_gradient(layer, max_norm=0.25):
    """Clip to avoid vanishing or exploding gradients

    :param layer: An instance of active layer
    :type layer:

    :param max_norm:
    :type max_norm: float
    """
    total_norm = 0

    # Calculate the L2 norm squared for each gradient and add them to the total norm
    for grad in layer.g.values():
        grad_norm = np.sum(np.power(grad, 2))
        total_norm += grad_norm

    total_norm = np.sqrt(total_norm)

    # Calculate clipping coeficient
    clip_coef = max_norm / (total_norm+1e-6)

    # If the total norm is larger than the maximum allowable norm, then clip the gradient
    if clip_coef < 1:
        for g in layer.g.keys():
            layer.g[g] *= clip_coef

    return None
