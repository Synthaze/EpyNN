#EpyNN/nnlibs/meta/models.py
import nnlibs.commons.plot as cp

import nnlibs.meta.forward as mf
import nnlibs.meta.train as mt


#@log_class
class EpyNN:

    def __init__(self,name='Model',layers=[]):
        self.n = name
        self.l = layers

    def predict(self,A):
        return mf.forward(self,A)

    def train(self,dsets,hPars,runData):
        mt.run_train(self,dsets,hPars,runData)

    def plot(self,hPars,runData):
        cp.pyplot_metrics(self,hPars,runData)
        cp.gnuplot_accuracy(runData)
