# EpyNN/depynn/commons/maths.py
# Related third party imports
import numpy as np


# To prevent from divide floatting points errors
E_SAFE = 1e-16

### Activation functions and derivatives

#

def _(x, deriv=False):
    """.

    .

    :param x: Input array to pass in function.
    :type x: class:`numpy.ndarray`

    :param deriv: To compute derivative, defaults to False.
    :type deriv: bool, optional

    :return: Output array passed in function.
    :rtype: :class:`numpy.ndarray`
    """
    if not deriv:
        pass

    elif deriv:
        pass

    return x
