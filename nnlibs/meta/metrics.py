# EpyNN/nnlibs/meta/metrics.py
# Related third party imports
import numpy as np

# Local application/library specific imports
from nnlibs.commons.loss import loss_functions


def model_compute_metrics(model):
    """Compute metrics for model.

    Will evaluate training, testing and validation sets against metrics set in model.se_config.

    :param model: An instance of EpyNN network.
    :type model: :class:`nnlibs.meta.models.EpyNN`
    """

    metrics = {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
    }

    metrics.update(loss_functions())

    dsets = model.embedding.dsets

    hPars = model.se_hPars

    for k, dset in enumerate(reversed(dsets)):

        X = dset.X

        dset.A = model.forward(X)

        for s in model.metrics.keys():

            m = metrics[s](dset.Y, dset.A)

            if m.ndim == 1:
                m = np.sum(m) / len(dset.ids)
            else:
                m = np.mean(m.mean(axis=1))

            model.metrics[s][len(dsets) - 1 - k].append(m)

    return None


def accuracy(Y, A):
    """Accuracy of prediction.
    """
    encoded = (Y.shape[1] > 1)

    P = np.argmax(A, axis=1) if encoded else np.around(A)
    y = np.argmax(Y, axis=1) if encoded else Y

    accuracy = (P - y == 0)

    return accuracy


def recall(Y, A):
    """Fraction of positive instances retrieved over the total.
    """
    encoded = (Y.shape[1] > 1)

    P = np.argmax(A, axis=1) if encoded else np.around(A)
    y = np.argmax(Y, axis=1) if encoded else y

    oy = P + y

    tp = np.sum(np.where(oy == 0,1,0))
    fp = np.sum(np.where(P == 0,1,0)) - tp
    tn = np.sum(np.where(oy == 2,1,0))
    fn = np.sum(P) - tn

    recall = (tp / (tp+fn))

    return recall


def precision(Y, A):
    """Fraction of positive samples among retrieved instances.
    """
    encoded = (Y.shape[1] > 1)

    P = np.argmax(A, axis=1) if encoded else np.around(A)
    y = np.argmax(Y, axis=1) if encoded else y

    oy = P + y

    tp = np.sum(np.where(oy == 0,1,0))
    fp = np.sum(np.where(P == 0,1,0)) - tp
    tn = np.sum(np.where(oy == 2,1,0))
    fn = np.sum(P) - tn

    precision = (tp / (tp+fp))

    return precision
