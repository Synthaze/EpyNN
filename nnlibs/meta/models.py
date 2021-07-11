#EpyNN/nnlibs/meta/models.py
from nnlibs.commons.models import runData, hPars
from nnlibs.commons.maths import seeding
from nnlibs.commons.decorators import *
import nnlibs.commons.plot as cp

import nnlibs.meta.forward as mf
import nnlibs.meta.train as mt

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
        hPars = self.hPars
        runData = self.runData
        mt.run_train(self,hPars,runData)


    def plot(self):
        """An example docstring for a method definition."""
        hPars = self.hPars
        runData = self.runData
        cp.pyplot_metrics(self,hPars,runData)
        cp.gnuplot_accuracy(runData)


    def predict(self,A):
        """An example docstring for a method definition."""
        return mf.forward(self,A)

    def logs(self):
        """An example docstring for a method definition."""
        for i, layer in enumerate(self.l):

            name = layer.__class__.__name__

            self.a.append({'Layer': name, 'Dimensions': [], 'FW_Shapes': [], 'BW_Shapes': [], 'Activation': [] })

            for attr, content in layer.__dict__.items():

                if attr == 'd':

                    for d, v in content.items():

                        self.a[-1]['Dimensions'].append(d+' = '+str(v))

                elif attr == 'fs':

                    for s, v in content.items():

                        self.a[-1]['FW_Shapes'].append(s+' = '+str(v))

                elif attr == 'bs':

                    for s, v in content.items():

                        self.a[-1]['BW_Shapes'].append(s+' = '+str(v))

                elif 'activate' in attr:

                    try:
                        gate = attr.split('_')[1]
                    except:
                        gate = 'input'

                    self.a[-1]['Activation'].append(gate+' = '+str(content.__name__))
