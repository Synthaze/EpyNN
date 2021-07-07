#EpyNN/nnlibs/template/parameters.py
import nnlibs.template.parameters as tp
import nnlibs.template.backward as tb
import nnlibs.template.forward as tf


class Template:

    def __init__(self):

        """ Layer attributes """
        ### Set layer init attribute to True
        self.init = True
        ### Set layer activation attributes
        self.activation = []
        tp.set_activation(self)
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

        ### Init shapes
        tp.init_shapes(self)


    def forward(self,A):
        # Forward pass
        A = tf.template_forward(self,A)
        return A


    def backward(self,dA):
        # Backward pass
        dA = tb.template_backward(self,dA)
        return dA
