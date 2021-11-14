# EpyNN/dpynn/commons/metrics.py
# Related third party imports
import numpy as np


def fscore(Y, A):
    """.

    :param Y: True labels for a set of samples.
    :type Y: :class:`numpy.ndarray`

    :param A: Output of forward propagation.
    :type A: :class:`numpy.ndarray`

    :return: F-score for each sample.
    :rtype: :class:`numpy.ndarray`
    """
    encoded = (Y.shape[1] > 1)    # Check if one-hot encoding of labels

    P = np.argmax(A, axis=1) if encoded else np.around(A)
    y = np.argmax(Y, axis=1) if encoded else Y

    oy = P + y

    tp = np.sum(np.where(oy == 0, 1, 0))        # True positive
    fp = np.sum(np.where(P == 0, 1, 0)) - tp    # False positive
    tn = np.sum(np.where(oy == 2, 1, 0))        # True negative
    fn = np.sum(P) - tn                         # False negative

    recall = (tp / (tp + 0.5*(fp+fn)))

    return recall
