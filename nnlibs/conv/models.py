#EpyNN/nnlibs/conv/models.py
from nnlibs.commons.decorators import *
import nnlibs.commons.maths as cm

import nnlibs.conv.parameters as cp
import nnlibs.conv.backward as cb
import nnlibs.conv.forward as cf


class Convolution:
    """
    Description for class

    :ivar var1: initial value: par1
    :ivar var2: initial value: par2
    """
    
    def __init__(self,n_filters,f_width,
            depth=1,
            stride=1,
            padding=0,
            activate=cm.relu,
            initialization=cm.xavier):

        """ Layer attributes """
        ### Set layer init attribute to True
        self.init = True
        ### Set layer weigths init attribute
        self.initialization = initialization
        ### Set layer activation attributes
        self.activation = [activate]
        cp.set_activation(self)
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
        self.attrs = ['X','Xt','A','Z','h','C','c','z','f','i','g','z','o']

        ### Init shapes
        cp.init_shapes(self,n_filters,f_width,depth,stride,padding)


    def forward(self,A):
        # Forward pass
        A = cf.convolution_forward(self,A)
        return A


    def backward(self,dA):
        # Backward pass
        dA = cb.convolution_backward(self,dA)
        return dA
