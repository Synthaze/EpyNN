#EpyNN/nnlibs/lstm/models.py
from nnlibs.commons.decorators import *
import nnlibs.commons.maths as cm

import nnlibs.lstm.backward as lb
import nnlibs.lstm.forward as lf

import numpy as np

#@log_class
class LSTM:

    def __init__(self,hidden_size,runData,vocab_size=None,output_size=None,activate_input=cm.sigmoid,activate_forget=cm.sigmoid,activate_memory=cm.tanh,activate_candidate=cm.tanh,activate_output=cm.sigmoid):

        self.init = True

        self.activate_input = activate_input
        self.derivative_input = cm.get_derivative(activate_input)

        self.activate_forget = activate_forget
        self.derivative_forget = cm.get_derivative(activate_forget)

        self.activate_memory = activate_memory
        self.activate_memory = cm.get_derivative(activate_memory)

        self.activate_candidate = activate_candidate
        self.derivative_candidate = cm.get_derivative(activate_candidate)

        self.activate_output = activate_output
        self.derivative_output = cm.get_derivative(activate_output)

        # Dimensions
        self.d = {}

        self.d['h'] = hidden_size

        if vocab_size == None:
            vocab_size = runData.e['v']

        self.d['v'] = vocab_size

        self.d['z'] = self.d['h'] + self.d['v']

        if output_size == None:
            output_size = vocab_size

        self.d['o'] = output_size

        # Shapes
        self.s = {}

        self.s['Wf'] = ( self.d['h'], self.d['z'] )
        self.s['Wi'] = ( self.d['h'], self.d['z'] )
        self.s['Wg'] = ( self.d['h'], self.d['z'] )
        self.s['Wo'] = ( self.d['h'], self.d['z'] )
        self.s['Wv'] = ( self.d['o'], self.d['h'] )

        self.s['bf'] = ( self.d['h'], 1 )
        self.s['bi'] = ( self.d['h'], 1 )
        self.s['bg'] = ( self.d['h'], 1 )
        self.s['bo'] = ( self.d['h'], 1 )
        self.s['bv'] = ( self.d['o'], 1 )

        # Parameters
        self.p = {}


        # Gradients
        self.g = {}


        # Cache
        self.c = {}

    def init_cache(self):

        for x in ['h','C','c','z','f','i','g','z','o','v']:

            self.c[x] = []

    def array_cache(self):

        for x in ['h','C','c','z','f','i','g','z','o','v']:

            self.c[x] = np.array(self.c[x])

    def forward(self,A):
        return lf.lstm_forward(self,A)

    def backward(self,dA):
        return lb.lstm_backward(self,dA)
