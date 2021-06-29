#EpyNN/nnlibs/conv/models.py
from nnlibs.commons.decorators import *
import nnlibs.commons.maths as cm

import nnlibs.conv.backward as cb
import nnlibs.conv.forward as cf


#@log_class
class Convolution:

    def __init__(self,num_filters,filter_width,activate,depth=1,stride=1,padding=0):

        self.init = True

        self.activate = activate
        self.derivative = cm.get_derivative(activate)

        # Dimensions
        self.d = {}

        self.d['fw'] = filter_width
        self.d['n_f'] = num_filters
        self.d['s'] = stride
        self.d['p'] = padding

        # Shapes
        self.s = {}

        self.s['W'] = (self.d['fw'], self.d['fw'], depth, self.d['n_f'])
        self.s['b'] = (1, 1, 1, self.d['n_f'])

        # Parameters
        self.p = {}

        # Gradients
        self.g = {}

    def forward(self,A):
        return cf.convolution_forward(self,A)

    def backward(self,dA):
        return cb.convolution_backward(self,dA)
