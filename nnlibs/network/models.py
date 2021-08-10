# EpyNN/nnlibs/network/models.py
# Standard library imports
import traceback
import time

# Related third party imports
import numpy as np

# Local application/library specific imports
from nnlibs.commons.io import encode_dataset
from nnlibs.commons.library import write_model
from nnlibs.commons.loss import loss_functions
from nnlibs.commons.models import dataSet
from nnlibs.network.report import (
    model_report,
    initialize_model_report,
)
from nnlibs.commons.plot import (
    pyplot_metrics,
    gnuplot_accuracy,
)
from nnlibs.network.initialize import (
    model_initialize,
    model_initialize_exceptions,
)
from nnlibs.network.hyperparameters import (
    model_hyperparameters,
    model_learning_rate,
)
from nnlibs.network.evaluate import model_evaluate
from nnlibs.network.forward import model_forward
from nnlibs.network.backward import model_backward
from nnlibs.network.training import model_training
from nnlibs.settings import se_hPars


class EpyNN:
    """
    Definition of a Neural Network prototype following the EpyNN scheme.

    :param layers: Network architecture
    :type layers: list[Object]

    :param name: Name of network, defaults to 'Network'
    :type name: str, optional
    """

    def __init__(self, layers, name='Network'):
        """Initialize instance variable attributes.

        :ivar layers: Network architecture
        :vartype layers: list[Object]

        :ivar embedding: Embedding layer
        :vartype embedding: :class:`nnlibs.embedding.models.Embedding`

        :ivar ts: Timestamp identifier
        :vartype ts: int

        :ivar uname: Network unique identifier
        :vartype uname: str

        :ivar initialized: Model initialization state
        :vartype initialized: bool
        """
        # Layers
        self.layers = layers
        self.embedding = self.layers[0]

        # Identification
        self.ts = int(time.time())
        self.uname = str(self.ts) + '_' + name

        # State
        self.initialized = False

        return None

    def forward(self, X):
        """Wrapper for :func:`nnlibs.network.forward.model_forward()`.

        :param X: Set of sample features.
        :type X: :class:`numpy.ndarray`

        :return: Output of forward propagation through all layers in the Network.
        :rtype: :class:`numpy.ndarray`
        """
        A = model_forward(self, X)

        return A

    def backward(self, dA):
        """Wrapper for :func:`nnlibs.network.backward.model_backward()`.

        :param dA: Derivative of ....
        :type dA: :class:`numpy.ndarray`

        :return: Output of forward propagation through all layers in the Network.
        :rtype: :class:`numpy.ndarray`
        """
        dA = model_backward(self, dA)

        return None

    def initialize(self, loss='MSE', se_hPars=se_hPars, metrics=['accuracy'], seed=None, params=True):
        """Initialize the Network.

        :param loss:
        :type loss: str

        :param se_hPars:
        :type se_hPars: dict[str: float or str]

        :param metrics:
        :type metrics: list[str]

        :param seed: For reproducibility in parameters initialization
        :type seed: int or NoneType, optional

        :param params: Layer parameters initialization.
        :type params: bool
        """
        self.training_loss = loss_functions(loss)
        self.se_hPars = se_hPars
        self.seed = seed

        model_hyperparameters(self)

        metrics.append(self.training_loss.__name__)
        self.metrics = {m:[[] for _ in range(3)] for m in metrics}

        try:
            model_initialize(self, params=params)
        except Exception:
            trace = traceback.format_exc()
            model_initialize_exceptions(self, trace)

        self.initialized = True

        return None

    def train(self, epochs, verbose=None, init_logs=True):
        """Wrapper for :func:`nnlibs.network.training.model_training()`.

        :param epochs: Number of training iterations
        :type epochs: dict[str: int or str]

        :param verbose: Print logs every Nth epochs, defaults to None which sets to every tenth of epochs.
        :type verbose: int or NoneType

        :param init_logs:
        :type init_logs: bool

        :param extend:
        :type extend: bool
        """
        if not self.initialized:
            self.initialize()

        self.epochs = epochs if not self.e else epochs + self.e + 1

        model_learning_rate(self)

        if init_logs:
            initialize_model_report(self, timeout=3)

        if not verbose:
            verbose = epochs // 10

        self.verbose = verbose

        model_training(self)

        return None

    def evaluate(self):
        """Wrapper for :func:`nnlibs.network.evaluate.model_evaluate()`.
        """
        model_evaluate(self)

        return None

    def write(self, model_path=None):
        """Write model on disk.

        :param model_path: ...
        :type model_path: str
        """
        write_model(self, model_path)

        return None

    def report(self):
        """Wrapper for :func:`nnlibs.network.report.model_report()`.
        """
        model_report(self)

        return None

    def plot(self, pyplot=True, gnuplot=False, path=None):
        """Plot metrics from model training.

        :param pyplot: Plot of results on GUI using matplotlib
        :type pyplot: bool

        :param gnuplot: Plot results on terminal using gnuplot
        :type gnuplot: bool

        :param path: Write matplotlib plot
        :type path: bool or NoneType
        """
        if pyplot:
            pyplot_metrics(self, path)

        if gnuplot:
            gnuplot_accuracy(self)

        return None

    def predict(self, X_data, X_encode=False, X_scale=False):
        """Perform prediction of label from unlabeled samples in dataset.

        :param dataset: Set of sample features.
        :type dataset: list[list[int or float or str]]

        :param X_encode: One-hot encode sample features.
        :type X_encode: bool

        :param X_encode: One-hot encode sample features.
        :type X_encode: bool

        :param X_scale: Normalize sample features within [0, 1]
        :type X_scale: bool

        :return: Data embedding and output of forward propagation
        :rtype: :class:`nnlibs.commons.models.dataSet`
        """
        if X_encode:
            word_to_idx = self.embedding.w2i
            vocab_size = self.embedding.d['v']
            X_data = encode_dataset(X_data, word_to_idx, vocab_size)

        if X_scale:
            X_data = scale_features(X_data)

        dset = dataSet(X_data)

        dset.A = self.forward(dset.X)

        dset.P = np.argmax(dset.A, axis=1)

        return dset
