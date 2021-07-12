#EpyNN/nnlibs/dropout/models.py
import nnlibs.dropout.parameters as dp
import nnlibs.dropout.backward as db
import nnlibs.dropout.forward as df

import numpy as np


class Dropout:
    """
    Description for class

    :ivar var1: initial value: par1
    :ivar var2: initial value: par2
    """

    def __init__(self,keep_prob=0.5):

        """ Layer attributes """
        ### Set layer init attribute to True
        self.init = True
        ### Define layer dictionaries attributes
        self.k = keep_prob
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
        self.attrs = ['X','D','A']

    def init_shapes(self):
        dp.init_shapes(self)

    def forward(self,A):
        A = df.dropout_forward(self,A)
        return A

    def backward(self,dA):
        dA = db.dropout_backward(self,dA)
        return dA
