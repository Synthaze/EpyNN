# EpyNN/nnlibs/commons/metrics.py
# Related third party imports
import numpy as np


def metrics_functions(key=None):

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
