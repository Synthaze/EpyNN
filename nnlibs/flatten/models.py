#EpyNN/nnlibs/flatten/parameters.py
import nnlibs.flatten.backward as fb
import nnlibs.flatten.forward as ff


class Flatten:
    """
    Description for class

    :ivar var1: initial value: par1
    :ivar var2: initial value: par2
    """
    
    def __init__(self):

        """ Layer attributes """
        ### Set layer init attribute to True
        self.init = True
        ### Set layer activation attributes
        self.activation = None
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
        self.attrs = ['X','A']


    def forward(self,A):
        # Forward pass
        A = ff.flatten_forward(self,A)
        return A

    def backward(self,dA):
        # Backward pass
        dA = fb.flatten_backward(self,dA)
        return dA
