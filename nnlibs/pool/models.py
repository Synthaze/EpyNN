#EpyNN/nnlibs/pool/models.py
import nnlibs.pool.backward as pb
import nnlibs.pool.forward as pf

import numpy as np


class Pooling:

    def __init__(self, filter_width, stride = 1):

        self.pool = np.max

        # Dimensions
        self.d = {}

        self.d['fw'] = filter_width
        self.d['s'] = stride

        # Shapes
        self.s = {}

    def forward(self,A):
        return pf.pooling_forward(self,A)

    def backward(self,dA):
        return pb.pooling_backward(self,dA)

    def assemble_block(self, block, t, b, l, r):

        ih = self.X.shape[1]
        iw = self.X.shape[2]

        block = np.repeat(block, self.d['fw'] ** 2, 2)

        block = np.array(np.split(block, block.shape[2] / self.d['fw'], 2))
        block = np.moveaxis(block, 0, 2)

        block = np.array(np.split(block, block.shape[2] / self.d['fw'], 2))
        block = np.moveaxis(block, 0, 3)

        return np.reshape(block, (self.X.shape[0], ih - t - b, iw - l - r, self.X.shape[3]))
