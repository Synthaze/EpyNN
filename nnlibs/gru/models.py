#EpyNN/nnlibs/gru/models.py
from nnlibs.commons.decorators import *
import nnlibs.commons.maths as cm

import nnlibs.gru.backward as gb
import nnlibs.gru.forward as gf

import numpy as np


#@log_class
class GRU:

    def __init__(self,hidden_size,runData,vocab_size=None,output_size=None,activate_input=cm.tanh,activate_update=cm.sigmoid,activate_reset=cm.sigmoid,activate_output=cm.softmax):

        self.init = True

        self.activate_input = activate_input
        self.derivative_input = cm.get_derivative(activate_input)

        self.activate_update = activate_update
        self.derivative_update = cm.get_derivative(activate_update)

        self.activate_reset = activate_reset
        self.derivative_reset = cm.get_derivative(activate_reset)

        self.activate_output = activate_output
        self.derivative_output = cm.get_derivative(activate_output)

        # Dimensions
        self.d = {}

        self.d['h'] = hidden_size

        if vocab_size == None:
            vocab_size = runData.e['v']

        self.d['v'] = vocab_size

        if output_size == None:
            output_size = vocab_size

        self.d['o'] = output_size

        # Shapes
        self.s = {}

        self.s['Wz'] = ( self.d['h'], self.d['v'] )
        self.s['Uz'] = ( self.d['h'], self.d['h'] )
        self.s['Wr'] = ( self.d['h'], self.d['v'] )
        self.s['Ur'] = ( self.d['h'], self.d['h'] )
        self.s['Wh'] = ( self.d['h'], self.d['v'] )
        self.s['Uh'] = ( self.d['h'], self.d['h'] )
        self.s['Wy'] = ( self.d['o'], self.d['h'] )

        self.s['bz'] = ( self.d['h'], 1 )
        self.s['br'] = ( self.d['h'], 1 )
        self.s['bh'] = ( self.d['h'], 1 )
        self.s['by'] = ( self.d['o'], 1 )

        # Parameters
        self.p = {}

        # Gradients
        self.g = {}

        # Caches
        self.c = {}

    def init_cache(self):

        for x in ['h','hp','z','r']:

            self.c[x] = []

    def array_cache(self):

        for x in ['h','hp','z','r']:

            self.c[x] = np.array(self.c[x])

    def forward(self,A):
        return gf.gru_forward(self,A)

    def backward(self,dA):
        return gb.gru_backward(self,dA)
