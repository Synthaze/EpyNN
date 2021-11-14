# EpyNN/epynn/commons/metrics.py
# Related third party imports
import numpy as np


def metrics_functions(key=None):
    """Callback function for metrics.

    :param key: Name of the metrics function, defaults to `None` which returns all functions.
    :type key: str, optional

    :return: Metrics functions or computed metrics.
    :rtype: dict[str: function] or :class:`numpy.ndarray`
    """
    metrics = {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'fscore': fscore,
        'specificity': specificity,
        'NPV': NPV,
    }
    # If key provided, returns output of function
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

    accuracy = (P == y)

    return accuracy


def recall(Y, A):
    """Fraction of positive instances retrieved over the total.

    :param Y: True labels for a set of samples.
    :type Y: :class:`numpy.ndarray`

    :param A: Output of forward propagation.
    :type A: :class:`numpy.ndarray`

    :return: Recall.
    :rtype: :class:`numpy.ndarray`
    """
    encoded = (Y.shape[1] > 1)    # Check if one-hot encoding of labels

    P = np.argmax(A, axis=1) if encoded else np.around(A)
    y = np.argmax(Y, axis=1) if encoded else Y

    tp = np.sum(np.where((P==0) & (y==0), 1, 0))    # True positive
    fp = np.sum(np.where((P==0) & (y==1), 1, 0))    # False positive
    tn = np.sum(np.where((P==1) & (y==1), 1, 0))    # True negative
    fn = np.sum(np.where((P==1) & (y==0), 1, 0))    # False negative

    recall = (tp / (tp+fn))

    return recall


def precision(Y, A):
    """Fraction of positive samples among retrieved instances.

    :param Y: True labels for a set of samples.
    :type Y: :class:`numpy.ndarray`

    :param A: Output of forward propagation.
    :type A: :class:`numpy.ndarray`

    :return: Precision.
    :rtype: :class:`numpy.ndarray`
    """
    encoded = (Y.shape[1] > 1)    # Check if one-hot encoding of labels

    P = np.argmax(A, axis=1) if encoded else np.around(A)
    y = np.argmax(Y, axis=1) if encoded else Y

    tp = np.sum(np.where((P==0) & (y==0), 1, 0))    # True positive
    fp = np.sum(np.where((P==0) & (y==1), 1, 0))    # False positive
    tn = np.sum(np.where((P==1) & (y==1), 1, 0))    # True negative
    fn = np.sum(np.where((P==1) & (y==0), 1, 0))    # False negative

    precision = (tp / (tp+fp))

    return precision


def NPV(Y, A):
    """Fraction of negative samples among excluded instances.

    :param Y: True labels for a set of samples.
    :type Y: :class:`numpy.ndarray`

    :param A: Output of forward propagation.
    :type A: :class:`numpy.ndarray`

    :return: Negative Predictive Value.
    :rtype: :class:`numpy.ndarray`
    """
    encoded = (Y.shape[1] > 1)    # Check if one-hot encoding of labels

    P = np.argmax(A, axis=1) if encoded else np.around(A)
    y = np.argmax(Y, axis=1) if encoded else Y

    tp = np.sum(np.where((P==0) & (y==0), 1, 0))    # True positive
    fp = np.sum(np.where((P==0) & (y==1), 1, 0))    # False positive
    tn = np.sum(np.where((P==1) & (y==1), 1, 0))    # True negative
    fn = np.sum(np.where((P==1) & (y==0), 1, 0))    # False negative

    npv = (tn / (tn+fn))

    return npv


def fscore(Y, A):
    """F-Score that is the harmonic mean of recall and precision.

    :param Y: True labels for a set of samples.
    :type Y: :class:`numpy.ndarray`

    :param A: Output of forward propagation.
    :type A: :class:`numpy.ndarray`

    :return: F-score.
    :rtype: :class:`numpy.ndarray`
    """
    encoded = (Y.shape[1] > 1)    # Check if one-hot encoding of labels

    P = np.argmax(A, axis=1) if encoded else np.around(A)
    y = np.argmax(Y, axis=1) if encoded else Y

    tp = np.sum(np.where((P==0) & (y==0), 1, 0))    # True positive
    fp = np.sum(np.where((P==0) & (y==1), 1, 0))    # False positive
    tn = np.sum(np.where((P==1) & (y==1), 1, 0))    # True negative
    fn = np.sum(np.where((P==1) & (y==0), 1, 0))    # False negative

    fscore = (tp / (tp + 0.5*(fp+fn)))

    return fscore

 
def specificity(Y, A):
    """Fraction of negative samples among excluded instances.

    :param Y: True labels for a set of samples.
    :type Y: :class:`numpy.ndarray`

    :param A: Output of forward propagation.
    :type A: :class:`numpy.ndarray`

    :return: Specificity.
    :rtype: :class:`numpy.ndarray`
    """
    encoded = (Y.shape[1] > 1)    # Check if one-hot encoding of labels

    P = np.argmax(A, axis=1) if encoded else np.around(A)
    y = np.argmax(Y, axis=1) if encoded else Y

    tp = np.sum(np.where((P==0) & (y==0), 1, 0))    # True positive
    fp = np.sum(np.where((P==0) & (y==1), 1, 0))    # False positive
    tn = np.sum(np.where((P==1) & (y==1), 1, 0))    # True negative
    fn = np.sum(np.where((P==1) & (y==0), 1, 0))    # False negative

    specificity = (tn / (tn+fp))

    return specificity
