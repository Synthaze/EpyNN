# EpyNN/nnlibs/meta/models.py
# Standard library imports
import traceback
import time

# Local application/library specific imports
from nnlibs.commons.metrics import model_compute_metrics
from nnlibs.commons.logs import model_logs, initialize_model_logs
from nnlibs.commons.plot import pyplot_metrics, gnuplot_accuracy
from nnlibs.meta.initialize import initialize_model_layers, initialize_exceptions
from nnlibs.meta.parameters import assign_seed_layers, compute_learning_rate
from nnlibs.meta.forward import model_forward
from nnlibs.meta.backward import model_backward
from nnlibs.meta.train import model_training
import nnlibs.settings as se


class EpyNN:
    """
    Definition of an . prototype
    """

    def __init__(self,
                layers=[],
                settings=[se.dataset, se.config, se.hPars],
                seed=None,
                name='Model'):

        self.ts = int(time.time())

        self.layers = layers
        self.embedding = self.layers[0]

        self.se_dataset, self.se_config, self.se_hPars = settings
        self.epochs = int(self.se_hPars['training_epochs'])

        self.seed = seed

        self.name = name
        self.uname = str(self.ts) + '_' + self.name

        self.network = { id(layer):{} for layer in self.layers }

        self.metrics = { m:[[] for _ in range(3)] for m in self.se_config['metrics_list'] }

        self.saved = False

        return None

    def forward(self, A):

        A = model_forward(self, A)
        return A

    def backward(self, dA):

        dA = model_backward(self, dA)
        return dA

    def initialize(self):
        assign_seed_layers(self)
        compute_learning_rate(self)

        try:
            initialize_model_layers(self)
        except Exception:
            trace = traceback.format_exc()
            initialize_exceptions(self,trace)

        initialize_model_logs(self)

        return None

    def train(self):

        model_training(self)

        return None

    def compute_metrics(self):

        model_compute_metrics(self)

        return None

    def evaluate(self):

        pass

        return None

    def logs(self):

        model_logs(self)

        return None

    def plot(self):

        pyplot_metrics(self)
        gnuplot_accuracy(self)

        return None

    # def embedding_unlabeled(self,X,encode=False):
    #
    #     if encode == True:
    #         embedding = self.l[0]
    #         X = ep.encode_dataset(embedding,X,unlabeled=True)
    #     else:
    #         X = [ x[0] for x in X ]
    #
    #     X = np.array(X)
    #
    #     return X
