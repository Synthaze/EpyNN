# EpyNN/epynn/network/evaluate.py
# Related third party imports
import numpy as np

# Local application/library specific imports
from epynn.commons.loss import loss_functions
from epynn.commons.metrics import metrics_functions


def model_evaluate(model):
    """Compute metrics including cost for model.

    Will evaluate training, testing and validation sets against metrics set in model.se_config.

    :param model: An instance of EpyNN network.
    :type model: :class:`epynn.network.models.EpyNN`
    """

    # Callback functions for metrics and loss
    metrics = metrics_functions()
    metrics.update(loss_functions())

    dsets = model.embedding.dsets

    # Iterate over dsets [dtrain, dval, dtest]
    for k, dset in enumerate(dsets):

        # Check if one-hot encoding
        encoded = (dset.Y.shape[1] > 1)

        # Output probs
        dset.A = model.forward(dset.X)

        # Decisions
        dset.P = np.argmax(dset.A, axis=1) if encoded else np.around(dset.A)

        # Iterate over selected metrics
        for s in model.metrics.keys():

            m = metrics[s](dset.Y, dset.A)

            # Metrics such as precision/recall returned as scalar
            if m.ndim == 0:
                pass
            # Others returned as per-sample 1D array
            else:
                m = np.mean(m)  # To scalar

            # Save value for metrics (s) for dset (k)
            model.metrics[s][k].append(m)

    return None


def batch_evaluate(model, Y, A):
    """Compute metrics for current batch.

    Will evaluate current batch against accuracy and training loss.

    :param model: An instance of EpyNN network.
    :type model: :class:`epynn.network.models.EpyNN`

    :param Y: True labels for batch samples.
    :type Y: :class:`numpy.ndarray`

    :param A: Output of forward propagation for batch.
    :type A: :class:`numpy.ndarray`
    """
    metrics = metrics_functions()

    # Per sample 1D array to scalar
    accuracy = np.mean(metrics['accuracy'](Y, A))

    # Per sample 1D array to scalar
    cost = np.mean(model.training_loss(Y, A))

    return accuracy, cost
