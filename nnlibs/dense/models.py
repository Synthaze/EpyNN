#EpyNN/nnlibs/dense/models.py
from nnlibs.commons.decorators import *

import nnlibs.dense.parameters as dp
import nnlibs.dense.backward as db
import nnlibs.dense.forward as df

import nnlibs.commons.maths as cm


class Dense:
    """
    Description for class

    :ivar var1: initial value: par1
    :ivar var2: initial value: par2
    """
    
    @log_method
    def __init__(self,
            layer_size=2,
            activate=cm.softmax,
            initialization=cm.xavier,
            l1 = 0,
            l2=0):

        """ Layer attributes """
        ### Set layer init attribute to True
        self.init = True
        ### Set layer weigths init attribute
        self.initialization = initialization
        ### Set layer activation attributes
        self.activation = [activate]
        dp.set_activation(self)
        ### Regularization l1
        self.l1 = l1
        ### Regularization l2
        self.l2 = l2
        ### Define layer dictionaries attributes
        ## Dimensions
        self.d = {}
        ## Parameters
        self.p = {}
        ## Gradients
        self.g = {}
        ## Forward pass cache
        self.fc = {}
        ## Backward pass cache
        self.bc = {}
        ## Forward pass shapes
        self.fs = {}
        ## Backward pass shapes
        self.bs = {}

        ### Set keys for layer cache attributes
        self.attrs = ['X','A','Z']

        ### Init shapes
        dp.init_shapes(self,layer_size)


    def forward(self,A):
        """ Layer attributes """
        # Forward pass
        A = df.dense_forward(self,A)
        return A


    def backward(self,dA):
        """ Layer attributes """
        # Backward pass
        dA = db.dense_backward(self,dA)
        return dA
