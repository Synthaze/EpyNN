# EpyNN/nnlibs/meta/models.py
# Standard library imports
import traceback
import time

# Local application/library specific imports
from nnlibs.commons.models import dataSet
from nnlibs.commons.io import encode_dataset
from nnlibs.commons.metrics import model_compute_metrics
from nnlibs.commons.library import write_model
from nnlibs.commons.logs import (
    model_logs,
    initialize_model_logs,
)
from nnlibs.commons.plot import (
    pyplot_metrics,
    gnuplot_accuracy,
)
from nnlibs.meta.initialize import (
    initialize_model_layers,
    initialize_exceptions,
)
from nnlibs.meta.parameters import (
    assign_seed_layers,
    compute_learning_rate,
)
from nnlibs.meta.forward import model_forward
from nnlibs.meta.backward import model_backward
from nnlibs.meta.training import model_training
from nnlibs.settings import (
    dataset as se_dataset,
    config as se_config,
    hPars as se_hPars,
)


class EpyNN:
    """
    Definition of an . prototype.
    """

    def __init__(self,
                layers=[],
                settings=[se_dataset, se_config, se_hPars],
                seed=None,
                name='Model'):
        """.

        :ivar ts:
        :vartype ts:
        :ivar layers:
        :vartype layers:
        :ivar embedding:
        :vartype embedding:
        :ivar se_dataset:
        :vartype se_dataset:
        :ivar se_config:
        :vartype se_config:
        :ivar se_hPars:
        :vartype se_hPars:
        :ivar epochs:
        :vartype epochs:
        :ivar seed:
        :vartype seed:
        :ivar name:
        :vartype name:
        :ivar uname:
        :vartype uname:
        :ivar network:
        :vartype network:
        :ivar metrics:
        :vartype metrics:
        :ivar saved:
        :vartype saved:
        """
        self.layers = layers
        self.se_dataset, self.se_config, self.se_hPars = settings
        self.seed = seed
        self.name = name

        self.ts = int(time.time())
        self.uname = str(self.ts) + '_' + self.name

        self.embedding = self.layers[0]

        self.network = {id(layer):{} for layer in self.layers}

        self.epochs = int(self.se_hPars['training_epochs'])

        self.metrics = {m:[[] for _ in range(3)] for m in self.se_config['metrics_list']}

        self.saved = False

        return None

    def forward(self, A):
        """.

        :param A:
        :type A:
        :return:
        :rtype:
        """
        A = model_forward(self, A)

        return A

    def backward(self, dA):
        """.

        :param dA:
        :type dA:
        :return:
        :rtype:
        """
        dA = model_backward(self, dA)

        return dA

    def initialize(self):
        """.
        """
        assign_seed_layers(self)
        compute_learning_rate(self)

        try:
            initialize_model_layers(self)
        except Exception:
            trace = traceback.format_exc()
            initialize_exceptions(self,trace)

        initialize_model_logs(self)

        return None

    def train(self, init=True, run=True):
        """.

        :param init:
        :type init:
        :param run:
        :type run:
        """
        if init:
            self.initialize()

        if run:
            model_training(self)

        return None

    def compute_metrics(self):
        """.
        """
        model_compute_metrics(self)

        return None

    def evaluate(self, write=False):
        """.

        :param write:
        :type write:
        """
        if write:
            write_model(self)

        return None

    def logs(self):
        """.
        """
        model_logs(self)

        return None

    def plot(self, pyplot=True, gnuplot=False):
        """.

        :param pyplot:
        :type pyplot:
        :param gnuplot:
        :type gnuplot:
        """
        if pyplot:
            pyplot_metrics(self)

        if gnuplot:
            gnuplot_accuracy(self)

        return None

    def predict(self, dataset, encode=False):
        """.

        :param dataset:
        :type dataset:
        :param encode:
        :type encode:
        :return:
        :rtype:
        """
        if encode:
            word_to_idx = self.embedding.w2i
            vocab_size = self.embedding.d['v']
            dataset = encode_dataset(dataset, word_to_idx, vocab_size)

        dset = dataSet(dataset, label=False)

        A = dset.X

        dset.A = self.forward(A).T

        return dset
