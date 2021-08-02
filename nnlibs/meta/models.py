# EpyNN/nnlibs/meta/models.py
# Standard library imports
import traceback
import time

# Local application/library specific imports
from nnlibs.commons.models import dataSet
from nnlibs.commons.io import encode_dataset
from nnlibs.commons.library import write_model
from nnlibs.commons.loss import loss_functions
from nnlibs.commons.logs import (
    model_logs,
    initialize_model_logs,
    start_counter,
)
from nnlibs.commons.plot import (
    pyplot_metrics,
    gnuplot_accuracy,
)
from nnlibs.meta.initialize import (
    model_initialize,
    model_initialize_exceptions,
)
from nnlibs.meta.parameters import (
    assign_seed_layers,
    compute_learning_rate,
)
from nnlibs.meta.metrics import model_compute_metrics
from nnlibs.meta.forward import model_forward
from nnlibs.meta.backward import model_backward
from nnlibs.meta.training import model_training


class EpyNN:
    """
    Definition of a Neural Network prototype following the EpyNN scheme.

    :param layers: Model architecture
    :type layers: list

    :param settings: Such as [se_dataset, se_config, se_hPars]
    :type settings: list[dict]

    :param seed: For model seeding
    :type seed: int or NoneType

    :param name: Name of model
    :type name: str
    """

    def __init__(self,
                layers,
                settings,
                seed=None,
                name='Model'):
        """Initialize instance variable attributes.

        :ivar ts: Timestamp unique identifier
        :vartype ts: int

        :ivar layers: Model architecture
        :vartype layers: list

        :ivar embedding: Embedding layer
        :vartype embedding: :class:`nnlibs.embedding.models.Embedding`

        :ivar se_dataset: Settings for dataset
        :vartype se_dataset: dict

        :ivar se_config: Settings for general configuration
        :vartype se_config: dict

        :ivar se_hPars: Settings for hyperparameters
        :vartype se_hPars: dict

        :ivar epochs: Number of training epochs
        :vartype epochs: int

        :ivar seed: For model seeding
        :vartype seed: int

        :ivar name: Name of model
        :vartype name: str

        :ivar uname: Unique name made of timestamp and name
        :vartype uname: str

        :ivar network: Store data related to network architecture
        :vartype network: dict

        :ivar metrics: Store metrics related to model training
        :vartype metrics: dict

        :ivar saved: Flag when model is saved on disk
        :vartype saved: bool
        """
        self.layers = layers
        self.se_dataset, self.se_config, self.se_hPars = settings
        self.seed = seed
        self.name = name

        self.ts = int(time.time())
        self.uname = str(self.ts) + '_' + self.name

        self.embedding = self.layers[0]

        self.network = {id(layer):{} for layer in self.layers}

        se_config = self.se_config

        self.epochs = se_config['training_epochs']
        self.e = 0

        self.training_loss = loss_functions(se_config['training_loss'])

        self.metrics = {m:[[] for _ in range(3)] for m in self.se_config['metrics_list']}

        self.saved = False

        return None

    def forward(self, X):
        """Wrapper for :func:`nnlibs.meta.forward.model_forward()`.
        """
        A = model_forward(self, X)

        return A

    def backward(self, dA):
        """Wrapper for :func:`nnlibs.meta.backward.model_backward()`.
        """
        dA = model_backward(self, dA)

        return dA

    def initialize(self, init_params=True, verbose=True):
        """Initialize EpyNN meta-model.
        """
        assign_seed_layers(self)
        compute_learning_rate(self)

        try:
            model_initialize(self, init_params=init_params)
        except Exception:
            trace = traceback.format_exc()
            model_initialize_exceptions(self,trace)

        if verbose:
            initialize_model_logs(self)
        else:
            self.current_logs = []

        return None

    def train(self, init=True, run=True):
        """Wrapper for :func:`nnlibs.meta.training.model_training()`.

        :param init: Skip model initialization if False
        :type init: bool

        :param run: Skip model training if False
        :type run: bool
        """
        if init:
            self.initialize()
            start_counter(timeout=3)

        if run:
            model_training(self)

        return None

    def compute_metrics(self):
        """Wrapper for :func:`nnlibs.meta.metrics.model_compute_metrics()`.
        """
        model_compute_metrics(self)

        return None

    def evaluate(self, write=False):
        """Evaluate model against metrics and write on disk accordingly.

        :param write: Arbitrarily write model on disk if True
        :type write: bool
        """
        if write:
            write_model(self)

        return None

    def logs(self):
        """Wrapper for :func:`nnlibs.commons.logs.model_logs()`.
        """
        model_logs(self)

        return None

    def plot(self, pyplot=True, gnuplot=False):
        """Plot metrics from model training.

        :param pyplot: Show plot of results using matplotlib
        :type pyplot: bool

        :param gnuplot: Show plot of results using gnuplot
        :type gnuplot: bool
        """
        if pyplot:
            pyplot_metrics(self)

        if gnuplot:
            gnuplot_accuracy(self)

        return None

    def predict(self, dataset, encode=False):
        """Perform prediction of label from unlabeled samples in dataset.

        :param dataset: Unlabeled samples in dataset
        :type dataset: list[list[list,list[int]]]

        :param encode: Perform one-hot encoding on features
        :type encode: bool

        :return: Data embedding and output of forward propagation
        :rtype: :class:`nnlibs.commons.models.dataSet`
        """
        if encode:
            word_to_idx = self.embedding.w2i
            vocab_size = self.embedding.d['v']
            dataset = encode_dataset(dataset, word_to_idx, vocab_size)

        dset = dataSet(dataset, label=False)

        X = dset.X

        dset.A = self.forward(X)

        return dset
