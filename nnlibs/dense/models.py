#EpyNN/nnlibs/dense/models.py
from nnlibs.commons.decorators import *

import nnlibs.dense.parameters as dp
import nnlibs.dense.backward as db
import nnlibs.dense.forward as df

import nnlibs.commons.maths as cm


class Dense:

    @log_method
    def __init__(self,
            layer_size=2,
            activate=cm.softmax):

        """ Layer attributes """
        ### Set layer init attribute to True
        self.init = True
        ### Set layer activation attributes
        self.activation = [activate]
        dp.set_activation(self)
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
        # Forward pass
        A = df.dense_forward(self,A)
        return A


    def backward(self,dA):
        # Backward pass
        dA = db.dense_backward(self,dA)
        return dA
