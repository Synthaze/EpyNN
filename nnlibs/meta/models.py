#EpyNN/nnlibs/meta/models.py
from nnlibs.commons.models import runData, hPars
from nnlibs.commons.maths import seeding
from nnlibs.commons.decorators import *
import nnlibs.commons.logs as clo
import nnlibs.commons.plot as cp

import nnlibs.meta.forward as mf
import nnlibs.meta.train as mt

import nnlibs.embedding.parameters as ep

import nnlibs.settings as se

import numpy as np


class EpyNN:
    """An example docstring for a class definition."""

    @log_method
    def __init__(self,name='Model',layers=[],settings=[se.dataset,se.config,se.hPars]):

        self.m = {}
        self.m['settings'] = settings

        self.n = name
        self.layers = self.l = layers

        self.runData = runData(settings[0],settings[1])

        self.hPars = hPars(settings[2])

        self.s = seeding()

        self.a = []

        self.g = {}

        embedding = layers[0]

        for i, layer in enumerate(self.l):

            layer.d['v'] = vocab_size = embedding.d['v']

            if self.s != None:
                layer.SEED = self.s + i
            else:
                layer.SEED = None

            layer.np = np.random.default_rng(seed=layer.SEED)

            layer.init_shapes()


    def train(self):
        """An example docstring for a method definition."""
        embedding = self.l[0]
        batch_dtrain = embedding.batch_dtrain
        mt.run_train(self,batch_dtrain)


    def embedding_unlabeled(self,X,encode=False):

        if encode == True:
            embedding = self.l[0]
            X = ep.encode_dataset(embedding,X,unlabeled=True)
        else:
            X = [ x[0] for x in X ]
            
        X = np.array(X)

        return X


    def predict(self,X):
        """An example docstring for a method definition."""
        A = X
        A = mf.forward(self,A)
        A = A.T
        return A

    def plot(self):
        """An example docstring for a method definition."""
        hPars = self.hPars
        runData = self.runData
        cp.pyplot_metrics(self,hPars,runData)
        cp.gnuplot_accuracy(runData)
