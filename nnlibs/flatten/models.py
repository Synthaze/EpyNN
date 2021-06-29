#EpyNN/nnlibs/flatten/parameters.py
import nnlibs.commons.maths as cm

import nnlibs.flatten.backward as fb
import nnlibs.flatten.forward as ff


class Flatten:

    def __init__(self,num_neurons=2,activate=cm.softmax):

        self.init = True

        self.activate = activate
        self.derivative = cm.get_derivative(activate)

        # Dimensions
        self.d = {}

        self.d['d'] = num_neurons

        # Shapes
        self.s = {}

        # Parameters
        self.p = {}

        # Gradients
        self.g = {}

    def forward(self,A):
        return ff.flatten_forward(self,A)

    def backward(self,dA):
        return fb.flatten_backward(self,dA)
