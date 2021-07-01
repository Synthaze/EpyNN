#EpyNN/nnlibs/gru/models.py
from nnlibs.commons.decorators import *
import nnlibs.commons.maths as cm

import nnlibs.gru.parameters as gp
import nnlibs.gru.backward as gb
import nnlibs.gru.forward as gf

import numpy as np


class GRU:

    def __init__(self,hidden_size,runData,
            output_size=None,
            activate_update=cm.sigmoid,
            activate_reset=cm.sigmoid,
            activate_input=cm.tanh,
            activate_output=cm.softmax):

        """ Layer attributes """
        ### Set layer init attribute to True
        self.init = True
        ### Set layer activation attributes
        self.activation = [activate_update,activate_reset,activate_input,activate_output]
        gp.set_activation(self)
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
        self.attrs = ['X','Xt','A','h','hp','z','r']

        ### Init shapes
        gp.init_shapes(self,hidden_size,runData)


    def forward(self,A):
        # Forward pass
        A = gf.gru_forward(self,A)
        return A

    def backward(self,dA):
        # Backward pass
        dA = gb.gru_backward(self,dA)
        return dA
