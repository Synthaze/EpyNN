#EpyNN/nnlibs/dense/models.py
import nnlibs.dense.backward as fb
import nnlibs.dense.forward as ff

import nnlibs.commons.maths as cm


class Dense:

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
        return ff.dense_forward(self,A)

    def backward(self,dA):
        return fb.dense_backward(self,dA)
