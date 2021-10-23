# EpyNN/epynn/network/hyperparameters.py
# Related third party imports
import numpy as np

# Local application/library specific imports
from epynn.commons.schedule import schedule_functions


def model_hyperparameters(model):
    """Set hyperparameters for each layer in model.

    :param model: An instance of EpyNN network.
    :type model: :class:`epynn.network.models.EpyNN`
    """
    # Iterate over layers
    for layer in model.layers:

        # If no se_hPars provided for layer, then assign se_hPars from model
        if not layer.se_hPars:
            layer.se_hPars = model.se_hPars
        else:
            pass

    return None


def model_learning_rate(model):
    """Schedule learning rate for each layer in model.

    :param model: An instance of EpyNN network.
    :type model: :class:`epynn.network.models.EpyNN`
    """
    # Iterate over layers
    for layer in model.layers:
        layer.se_hPars, layer.lrate = schedule_lrate(layer.se_hPars, model.epochs)

    return None


def schedule_lrate(se_hPars, training_epochs):
    """Learning rate schedule.

    :param se_hPars: Hyperparameters settings for layer.
    :type se_hPars: dict

    :param training_epochs: Number of training epochs for model.
    :type training_epochs: int

    :return: Updated settings for layer hyperparameters.
    :rtype: dict

    :return: Scheduled learning rate for layer.
    :rtype: list
    """
    # Extract hyperparameters
    e = se_hPars['epochs'] = training_epochs
    lr = se_hPars['learning_rate']
    d = se_hPars['cycle_descent']
    k = se_hPars['decay_k']

    # Compute dependent hyperparameters
    epc = se_hPars['cycle_epochs'] if se_hPars['cycle_epochs'] else training_epochs
    se_hPars['cycle_number'] = n = 1 if not epc else e // epc

    # Default decay
    if k == 0:
        # ~ 1% of initial lr for last epoch in cycle
        k = se_hPars['decay_k'] = 5 / epc

    # Compute learning rate schedule
    hPars = (e, lr, n, k, d, epc)
    lrate = schedule_functions(se_hPars['schedule'], hPars)

    return se_hPars, lrate
