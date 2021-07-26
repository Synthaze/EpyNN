# EpyNN/nnlibs/meta/models.py
# Standard library imports
import traceback

# Related third party imports
import numpy as np

# Local application/library specific imports
from nnlibs.commons.metrics import model_compute_metrics
from nnlibs.commons.logs import model_logs, initialize_model_logs
import nnlibs.commons.plot as cp

from nnlibs.meta.initialize import initialize_model_layers, initialize_exceptions
from nnlibs.meta.parameters import assign_seed_layers, compute_learning_rate

from nnlibs.meta.forward import model_forward
from nnlibs.meta.backward import model_backward
import nnlibs.meta.train as mt

import nnlibs.embedding.parameters as ep

import nnlibs.settings as se


class EpyNN:

    def __init__(self, layers=[], settings=se, seed=None, name='Model'):

        self.layers = layers

        self.se_dataset = settings.dataset
        self.se_config = settings.config
        self.se_hPars = settings.hPars

        self.seed = seed

        self.name = name

        self.network = { id(layer):{} for layer in self.layers }

        self.metrics = { m:[[]]*3 for m in self.se_config['metrics_list'] }

    def initialize(self):

        assign_seed_layers(self)

        try:
            initialize_model_layers(self)
            initialize_model_logs(self)

        except Exception:
            trace = traceback.format_exc()
            initialize_exceptions(self,trace)

        return None

    def train(self):
        """An example docstring for a method definition."""
        embedding = self.layers[0]
        batch_dtrain = embedding.batch_dtrain
        mt.run_train(self,batch_dtrain)

    def forward(self, A):
        A = model_forward(self, A)
        return A

    def backward(self, dA):
        dA = model_backward(self, dA)
        return dA

    def compute_metrics(self):
        model_compute_metrics(self)
        return None

    def evaluate(self):
        pass
        return None

    def logs(self):
        model_logs(self)
        return None

    def embedding_unlabeled(self,X,encode=False):

        if encode == True:
            embedding = self.l[0]
            X = ep.encode_dataset(embedding,X,unlabeled=True)
        else:
            X = [ x[0] for x in X ]

        X = np.array(X)

        return X


    def plot(self):
        """An example docstring for a method definition."""
        hPars = self.hPars
        runData = self.runData
        cp.pyplot_metrics(self,hPars,runData)
        cp.gnuplot_accuracy(runData)
