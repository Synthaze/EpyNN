#EpyNN/nnlibs/rnn/models.py
from nnlibs.commons.decorators import *
import nnlibs.commons.maths as cm

import nnlibs.rnn.parameters as rp

import nnlibs.rnn.backward as rb
import nnlibs.rnn.forward as rf


#@log_class
class RNN:

    def __init__(self,hidden_size,runData,
            binary=False,
            activate_input=cm.tanh,
            activate_output=cm.softmax,
            initialization=cm.xavier):

        """ Layer attributes """
        ### Set layer init attribute to True
        self.init = True
        ### Set layer weigths init attribute
        self.initialization = initialization
        ### Set layer activation attributes
        self.activation = [activate_input,activate_output]
        rp.set_activation(self)
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
        self.attrs = ['X','Xt','A','h']

        ### Init shapes
        self.binary = binary
        rp.init_shapes(self,hidden_size,runData)


    def forward(self,A):
        # Forward pass
        A = rf.rnn_forward(self,A)
        return A

    def backward(self,dA):
        # Backward pass
        dA = rb.rnn_backward(self,dA)
        return dA
