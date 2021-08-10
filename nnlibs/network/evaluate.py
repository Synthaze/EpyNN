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

        for s in model.metrics.keys():

            m = metrics[s](dset.Y, dset.A)

            if m.ndim == 0:
                pass

            elif m.ndim == 1:
                m = np.sum(m) / len(dset.ids)

            else:
                m = np.mean(m.mean(axis=1))

            model.metrics[s][k].append(m)

    return None
