#EpyNN/nnlibs/lstm/models.py
from nnlibs.commons.decorators import *
import nnlibs.commons.maths as cm

import nnlibs.lstm.parameters as lp
import nnlibs.lstm.backward as lb
import nnlibs.lstm.forward as lf

import numpy as np


class LSTM:

    @log_method
    def __init__(self,hidden_size,runData,
            binary=False,
            activate_forget=cm.sigmoid,
            activate_input=cm.sigmoid,
            activate_candidate=cm.tanh,
            activate_output=cm.sigmoid,
            activate_hidden=cm.tanh,
            activate_logits=cm.softmax,
            initialization=cm.xavier):

        """ Layer attributes """
        ### Set layer init attribute to True
        self.init = True
        ### Set layer weigths init attribute
        self.initialization = initialization
        ### Set layer activation attributes
        self.activation = [activate_forget,activate_input,activate_candidate,activate_output,activate_hidden,activate_logits]
        lp.set_activation(self)
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
        self.attrs = ['X','Xt','A','Z','h','c','z','f','i','g','z','o']

        ### Init shapes
        self.binary = binary
        lp.init_shapes(self,hidden_size,runData)


    def forward(self,A):
        # Forward pass
        A = lf.lstm_forward(self,A)
        return A

    def backward(self,dA):
        # Backward pass
        dA = lb.lstm_backward(self,dA)
        return dA
