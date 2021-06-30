#EpyNN/nnlibs/commons/maths.py
from nnlibs.commons.decorators import *

import numpy as np
import random


# Seeding
@log_function
def global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    global SEED
    SEED = seed


@log_function
def seeding():
    try: return SEED
    except: return None


# Constant parameters
@log_function
def global_constant(hPars):
    global CST
    CST = hPars.c


# Functions
def relu(x):
    return np.maximum(0,x)


def lrelu(x):
    a = CST['l']
    return np.maximum(a*x, x)


def elu(x):
    a = CST['e']
    return np.where(x>0, x, a*(np.exp(x,where=x<=0)-1))


def swish(x):
    return x/(1-np.exp(-x))


def sigmoid(x):
    z = (1/(1 + np.exp(-x)))
    return z


def tanh(x):
    z = (2/(1 + np.exp(-2*x))) - 1
    return z


def softmax(x):
    T = CST['s']
    x = x - np.max(x,axis=1,keepdims=True)
    x = np.exp(x/T)
    x_ = np.sum(x,axis=0,keepdims=True)
    x_ = x / x_
    return x_


# Derivatives

def drelu(dA,x):
    return dA * np.greater(x, 0).astype(int)


def delu(dA,x):
    a = CST['e']
    
    x = np.where(x>0, 1, elu(x)+a)

    dZ = dA * x

    return dZ


def dlrelu(dA,x):
    a = CST['l']
    return dA * np.where(x>0, 1, a)


def dtanh(dA,x):
    dZ = 1 - np.square(tanh(x))
    return dA * dZ


def dsigmoid(dA,x):
    a = sigmoid(x)
    return dA * a * (1 - a)


def dsoftmax(dA,x):
    return dA / (1.0 + np.exp(-x)) * (1.0 - (1.0 / (1.0 + np.exp(-x))))


# Utils

def get_derivative(activate):

    DFUNCS = {
        relu: drelu,
        lrelu: dlrelu,
        elu: delu,
        sigmoid: dsigmoid,
        softmax: dsoftmax,
        tanh: dtanh
        }

    return DFUNCS[activate]
