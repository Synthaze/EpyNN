#EpyNN/nnlibs/pool/models.py
import nnlibs.pool.parameters as pp
import nnlibs.pool.backward as pb
import nnlibs.pool.forward as pf

import numpy as np


class Pooling:

    def __init__(self,f_width,
            stride=1,
            pool=np.max):

        """ Layer attributes """
        ### Set layer init attribute to True
        self.init = True
        ### Set layer activation attributes
        self.pool = pool

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
        self.attrs = ['X','A','Z']

        ### Init shapes
        pp.init_shapes(self,f_width,stride)

    def assemble_block(self, block, t, b, l, r):

        im, ih, iw, id = self.fs['X']

        block = np.repeat(block, self.d['fw'] ** 2, 2)

        block = np.array(np.split(block, block.shape[2] / self.d['fw'], 2))
        block = np.moveaxis(block, 0, 2)

        block = np.array(np.split(block, block.shape[2] / self.d['fw'], 2))
        block = np.moveaxis(block, 0, 3)

        return np.reshape(block, ( im, ih - t - b, iw - l - r,  id))


    def forward(self,A):
        # Forward pass
        A = pf.pooling_forward(self,A)
        return A

    def backward(self,dA):
        # Backward pass
        dA = pb.pooling_backward(self,dA)
        return dA
