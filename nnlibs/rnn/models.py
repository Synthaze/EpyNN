#EpyNN/nnlibs/rnn/models.py
from nnlibs.commons.decorators import *

import nnlibs.rnn.backward as rb
import nnlibs.rnn.forward as rf


#@log_class
class RNN:

    def __init__(self,hidden_size,runData,vocab_size=None,output_size=None,activate_input=cm.tanh,activate_output=cm.softmax):

        self.init = True

        self.activate_input = activate_input
        self.derivative_input = cm.get_derivative(activate_input)

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

        self.s['U'] = ( self.d['h'], self.d['v'] )
        self.s['V'] = ( self.d['h'], self.d['h'] )
        self.s['W'] = ( self.d['o'], self.d['h'] )

        self.s['bh'] = ( self.d['h'], 1 )
        self.s['bo'] = ( self.d['o'], 1 )

        # Parameters
        self.p = {}

        # Gradients
        self.g = {}


    def forward(self,A):
        return rf.rnn_forward(self,A)

    def backward(self,dA):
        return rb.rnn_backward(self,dA)
