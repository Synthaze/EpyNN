#EpyNN/nnlibs/meta/models.py
from nnlibs.commons.maths import seeding
from nnlibs.commons.decorators import *
import nnlibs.commons.plot as cp

import nnlibs.meta.forward as mf
import nnlibs.meta.train as mt

import numpy as np


class EpyNN:
    """An example docstring for a class definition."""

    @log_method
    def __init__(self,name='Model',layers=[],hPars=None):

        self.n = name
        self.l = layers

        self.s = seeding()

        self.a = []

        self.g = {}

        for i, layer in enumerate(self.l):

            if self.s != None:
                layer.SEED = self.s + i
            else:
                layer.SEED = None

            layer.np = np.random.default_rng(seed=layer.SEED)


    def train(self,dsets,hPars,runData):
        """An example docstring for a method definition."""
        mt.run_train(self,dsets,hPars,runData)


    def plot(self,hPars,runData):
        """An example docstring for a method definition."""
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
