#EpyNN/nnlibs/dropout/models.py
import nnlibs.dropout.backward as db
import nnlibs.dropout.forward as df

import numpy as np


class Dropout:

    def __init__(self,keep_prob=0.5):

        self.init = True

        self.k = keep_prob

        # Parameters
        self.p = {}

        # Gradients
        self.g = {}

        # Shapes
        self.s = {}


    def forward(self,A):
        return df.dropout_forward(self,A)

    def backward(self,dA):
        return db.dropout_backward(self,dA)
