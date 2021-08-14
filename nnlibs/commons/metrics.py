# EpyNN/nnlibs/commons/metrics.py
# Related third party imports
import numpy as np


def metrics_functions(key=None):
    """Callback function for metrics.

    :param key: Name of the metrics function.
    :type key: str, optional

    :return: Metrics functions or computed metrics.
    :rtype: dict[str: function] or :class:`numpy.ndarray`
    """
    metrics = {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
    }

    if key:
        metrics = metrics[key]

    return metrics


def accuracy(Y, A):
    """Accuracy of prediction.

    :param Y: True labels for a set of samples.
    :type Y: :class:`numpy.ndarray`

    :param A: Output of forward propagation.
    :type A: :class:`numpy.ndarray`

    :return: Accuracy for each sample.
    :rtype: :class:`numpy.ndarray`
    """
    encoded = (Y.shape[1] > 1)

    P = np.argmax(A, axis=1) if encoded else np.around(A)
    y = np.argmax(Y, axis=1) if encoded else Y

    accuracy = (P - y == 0)

    return accuracy


def recall(Y, A):
    """Fraction of positive instances retrieved over the total.

    :param Y: True labels for a set of samples.
    :type Y: :class:`numpy.ndarray`

    :param A: Output of forward propagation.
    :type A: :class:`numpy.ndarray`

    :return: Recall for each sample.
    :rtype: :class:`numpy.ndarray`
    """
    encoded = (Y.shape[1] > 1)

    P = np.argmax(A, axis=1) if encoded else np.around(A)
    y = np.argmax(Y, axis=1) if encoded else y

    oy = P + y

    tp = np.sum(np.where(oy == 0, 1, 0))
    fp = np.sum(np.where(P == 0, 1, 0)) - tp
    tn = np.sum(np.where(oy == 2, 1, 0))
    fn = np.sum(P) - tn

    recall = (tp / (tp+fn))

    return recall


def precision(Y, A):
    """Fraction of positive samples among retrieved instances.

    :param Y: True labels for a set of samples.
    :type Y: :class:`numpy.ndarray`

    :param A: Output of forward propagation.
    :type A: :class:`numpy.ndarray`

    :return: Precision for each sample.
    :rtype: :class:`numpy.ndarray`
    """
    encoded = (Y.shape[1] > 1)

    P = np.argmax(A, axis=1) if encoded else np.around(A)
    y = np.argmax(Y, axis=1) if encoded else y

    oy = P + y

    tp = np.sum(np.where(oy == 0, 1, 0))
    fp = np.sum(np.where(P == 0, 1, 0)) - tp
    tn = np.sum(np.where(oy == 2, 1, 0))
    fn = np.sum(P) - tn

    precision = (tp / (tp+fp))

    return precision
