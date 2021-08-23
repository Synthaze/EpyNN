# EpyNN/nnlibs/network/evaluate.py
# Related third party imports
import numpy as np

# Local application/library specific imports
from nnlibs.commons.loss import loss_functions
from nnlibs.commons.metrics import metrics_functions


def model_evaluate(model):
    """Compute metrics for model.

    Will evaluate training, testing and validation sets against metrics set in model.se_config.

    :param model: An instance of EpyNN network.
    :type model: :class:`nnlibs.network.models.EpyNN`
    """

    metrics = metrics_functions()

    metrics.update(loss_functions())

    dsets = model.embedding.dsets

    for k, dset in enumerate(dsets):

        dset.A = model.forward(dset.X)

        dset.P = np.argmax(dset.A, axis=1)

        for s in model.metrics.keys():

            m = metrics[s](dset.Y, dset.A)

            if m.ndim == 0:
                pass

            else:
                m = np.mean(m)

            model.metrics[s][k].append(m)

    return None


def batch_evaluate(model, Y, A):
    """Compute metrics for model.

    Will evaluate training, testing and validation sets against metrics set in model.se_config.

    :param model: An instance of EpyNN network.
    :type model: :class:`nnlibs.network.models.EpyNN`
    """
    metrics = metrics_functions()

    metrics.update(loss_functions())

    accuracy = np.sum(metrics['accuracy'](Y, A)) / Y.shape[0]

    cost = np.mean(model.training_loss(Y, A))

    return accuracy, cost
