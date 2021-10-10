# EpyNN/epynn/network/models.py
# Standard library imports
import traceback
import time

# Related third party imports
import numpy as np

# Local application/library specific imports
from epynn.commons.io import (
    encode_dataset,
    scale_features,
)
from epynn.commons.library import write_model
from epynn.commons.loss import loss_functions
from epynn.commons.models import dataSet
from epynn.network.report import (
    model_report,
    initialize_model_report,
    single_batch_report,
)
from epynn.commons.plot import pyplot_metrics
from epynn.network.initialize import (
    model_assign_seeds,
    model_initialize,
    model_initialize_exceptions,
)
from epynn.network.hyperparameters import (
    model_hyperparameters,
    model_learning_rate,
)
from epynn.network.evaluate import model_evaluate
from epynn.network.forward import model_forward
from epynn.network.backward import model_backward
from epynn.network.training import model_training
from epynn.settings import se_hPars


class EpyNN:
    """
    Definition of a Neural Network prototype following the EpyNN scheme.

    :param layers: Network architecture.
    :type layers: list[Object]

    :param name: Name of network, defaults to 'EpyNN'.
    :type name: str, optional
    """

    def __init__(self, layers, name='EpyNN'):
        """Initialize instance variable attributes.

        :ivar layers: Network architecture.
        :vartype layers: list[Object]

        :ivar embedding: Embedding layer.
        :vartype embedding: :class:`epynn.embedding.models.Embedding`

        :ivar ts: Timestamp identifier.
        :vartype ts: int

        :ivar uname: Network unique identifier.
        :vartype uname: str

        :ivar initialized: Model initialization state.
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
        """Wrapper for :func:`epynn.network.forward.model_forward()`.

        :param X: Set of sample features.
        :type X: :class:`numpy.ndarray`

        :return: Output of forward propagation through all layers in the Network.
        :rtype: :class:`numpy.ndarray`
        """
        A = model_forward(self, X)

        return A

    def backward(self, dA):
        """Wrapper for :func:`epynn.network.backward.model_backward()`.

        :param dA: Derivative of the loss function with respect to the output of forward propagation.
        :type dA: :class:`numpy.ndarray`
        """
        dX = model_backward(self, dA)

        return None

    def initialize(self, loss='MSE', se_hPars=se_hPars, metrics=['accuracy'], seed=None, params=True, end='\n'):
        """Wrapper for :func:`epynn.network.initialize.model_initialize()`. Perform a dry epoch including all but not the parameters update step.

        :param loss: Loss function to use for training, defaults to 'MSE'. See :py:mod:`epynn.commons.loss` for built-in functions.
        :type loss: str, optional

        :param se_hPars: Global hyperparameters, defaults to :class:`epynn.settings.se_hPars`. If local hyperparameters were assigned to one layer, these remain unchanged.
        :type se_hPars: dict[str: float or str], optional

        :param metrics: Metrics to monitor and print on terminal report or plot, defaults to ['accuracy']. See :py:mod:`epynn.commons.metrics` for built-in metrics. Note that it also accept loss functions string identifiers.
        :type metrics: list[str], optional

        :param seed: Reproducibility in pseudo-random procedures.
        :type seed: int or NoneType, optional

        :param params: Layer parameters initialization, defaults to `True`.
        :type params: bool, optional

        :param end: Whether to print every line for initialization steps or overwrite, default to `\\n`.
        :type end: str in ['\\n', '\\r'], optional
        """
        # Initialize model summary
        self.network = {id(layer):{} for layer in self.layers}

        # Initialize storage for selected metrics evaluation
        metrics = metrics.copy()
        metrics.append(loss)

        self.metrics = {m:[[] for _ in range(3)] for m in metrics}

        # Check consistency output activation and loss
        self.output = self.layers[-1].activation['activate']
        self.training_loss = loss_functions(loss, self.output)

        # Assign model and layers hyperparameters
        self.se_hPars = se_hPars
        model_hyperparameters(self)

        # Seed model and layers
        self.seed = seed
        model_assign_seeds(self)

        try:
            # Attempt to initialize model
            model_initialize(self, params=params, end=end)

        except Exception:
            # Handle errors and provide debug info
            trace = traceback.format_exc()
            model_initialize_exceptions(self, trace)

        # Termination
        self.initialized = True

        return None

    def train(self, epochs, verbose=None, init_logs=True):
        """Wrapper for :func:`epynn.network.training.model_training()`. Apart, it computes learning rate along learning epochs.

        :param epochs: Number of training iterations.
        :type epochs: int

        :param verbose: Print logs every Nth epochs, defaults to `None` which sets to every tenth of epochs.
        :type verbose: int or NoneType, optional

        :param init_logs: Print data, architecture and hyperparameters logs, defaults to `True`.
        :type init_logs: bool, optional
        """
        # Model initialization
        if not self.initialized:
            self.initialize()

        # Handling training intiation or continuation scenarii
        self.epochs = epochs if not self.e else epochs + self.e + 1

        # Compute learning rate schedule for layers in model
        model_learning_rate(self)

        if init_logs:
            # From model.initialize() method
            initialize_model_report(self, timeout=3)

        if not verbose:
            # By defaut, store full evaluation one every tenth of epochs
            verbose = epochs // 10 if epochs >= 10 else 1

        # Start training
        self.verbose = verbose
        self.cts = time.time()

        model_training(self)

        return None

    def evaluate(self):
        """Wrapper for :func:`epynn.network.evaluate.model_evaluate()`. Good spot for further implementation of early stopping procedures.
        """
        model_evaluate(self)

        return None

    def write(self, path=None):
        """Write model on disk.

        :param path: Path to write the model on disk, defaults to `None` which writes in the `models` subdirectory created from :func:`epynn.commons.library.configure_directory()`.
        :type path: str or NoneType, optional
        """
        write_model(self, path)

        return None

    def batch_report(self, batch, A):
        """Wrapper for :func:`epynn.network.report.single_batch_report()`.
        """
        single_batch_report(self, batch, A)

        return None

    def report(self):
        """Wrapper for :func:`epynn.network.report.model_report()`.
        """
        model_report(self)

        return None

    def plot(self, pyplot=True, path=None):
        """Wrapper for :func:`epynn.commons.plot.pyplot_metrics()`. Plot metrics from model training.

        :param pyplot: Plot of results on GUI using matplotlib.
        :type pyplot: bool, optional

        :param path: Write matplotlib plot, defaults to `None` which writes in the `plots` subdirectory created from :func:`epynn.commons.library.configure_directory()`. To not write the plot at all, set to `False`.
        :type path: str or bool or NoneType, optional
        """
        if pyplot:
            pyplot_metrics(self, path)

        return None

    def predict(self, X_data, X_encode=False, X_scale=False):
        """Perform prediction of label from unlabeled samples in dataset.

        :param X_data: Set of sample features.
        :type X_data: list[list[int or float or str]] or :class:`numpy.ndarray`

        :param X_encode: One-hot encode sample features, defaults to `False`.
        :type X_encode: bool, optional

        :param X_scale: Normalize sample features within [0, 1] along all axis, default to `False`.
        :type X_scale: bool, optional

        :return: Data embedding and output of forward propagation.
        :rtype: :class:`epynn.commons.models.dataSet`
        """
        X_data = np.array(X_data)

        if X_encode:
            # One-hot encoding using embedding layer cache
            element_to_idx = self.embedding.e2i
            elements_size = self.embedding.d['e']
            X_data = encode_dataset(X_data, element_to_idx, elements_size)

        if X_scale:
            # Array-wide normalization in [0, 1]
            X_data = scale_features(X_data)

        dset = dataSet(X_data)

        # Predict
        dset.A = self.forward(dset.X)

        # Check label encoding
        encoded = (self.embedding.dtrain.Y.shape[1] > 1)

        # Make decisions
        dset.P = np.argmax(dset.A, axis=1) if encoded else np.around(dset.A)

        return dset
