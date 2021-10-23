# EpyNN/epynn/network/report.py
# Standard library imports
import time

# Related third party imports
from tabulate import tabulate
from termcolor import cprint

# Local application/library specific imports
from epynn.network.evaluate import batch_evaluate
from epynn.commons.logs import (
    current_logs,
    dsets_samples_logs,
    dsets_labels_logs,
    headers_logs,
    initialize_logs_print,
    layers_lrate_logs,
    layers_others_logs,
    network_logs,
    start_counter
)


def model_report(model):
    """Report selected metrics for datasets at current epoch.

    :param model: An instance of EpyNN network object.
    :type model: :class:`epynn.network.models.EpyNN`
    """
    # You may edit the colorscheme to fulfill your preference
    colors = [
        'white',
        'green',
        'red',
        'magenta',
        'cyan',
        'yellow',
        'blue',
        'grey',
    ]

    # Rows in tabular report excluding headers
    size_table = 11

    # Initialize list of rows with headers
    if model.e == 0 or not hasattr(model, 'current_logs'):
        model.current_logs = [headers_logs(model, colors)]

    # Check if last epoch
    eLast = (model.e == model.epochs - 1)

    # Append row one every verboseth epoch or if last epoch
    if model.e % model.verbose == 0 or eLast:
        model.current_logs.append(current_logs(model, colors))

    # Report on terminal
    if len(model.current_logs) == size_table + 1 or eLast:

        logs = tabulate(model.current_logs,
                        headers="firstrow",
                        numalign="center",
                        stralign='center',
                        tablefmt="pretty",
                        )

        print('\n')
        print (logs, flush=True)

        # Clear-up
        del model.current_logs

    return None


def single_batch_report(model, batch, A):
    """Report accuracy and cost for current batch.

    :param model: An instance of EpyNN network.
    :type model: :class:`epynn.network.models.EpyNN`

    :param batch: An instance of batch dataSet.
    :type batch: :class:`epynn.commons.models.dataSet`

    :param A: Output of forward propagation for batch.
    :type A: :class:`numpy.ndarray`
    """
    current = time.time()

    # Total elapsed time
    elapsed_time = round(current - model.ts, 2)

    # Time for one epoch based on current batch
    epoch_time = (current - model.cts) * len(model.embedding.batch_dtrain)
    model.cts = current

    # Epochs per second
    rate = round((model.e + 1) / (elapsed_time + 1e-16), 3)

    # Time until completion
    ttc = round((model.epochs - model.e + 1) / (rate + 1e-16))

    # Accuracy and cost
    accuracy, cost = batch_evaluate(model, batch.Y, A)
    accuracy = round(accuracy, 3)
    cost = round(cost, 5)

    # Current batch numerical identifier
    batch_counter = batch.name + '/' + model.embedding.batch_dtrain[-1].name

    # Format and print data
    rate = '{:.2e}'.format(rate)

    log = ('Epoch %s - Batch %s - Accuracy: %s Cost: %s - TIME: %ss RATE: %se/s TTC: %ss'
           % (model.e, batch_counter, accuracy, cost, elapsed_time, rate, ttc))

    cprint('{: <100}'.format(log), 'white', attrs=['bold'], end='\r', flush=True)

    return None


def initialize_model_report(model, timeout):
    """Report exhaustive initialization logs for datasets,
    model architecture and shapes, layers hyperparameters.

    :param model: An instance of EpyNN network.
    :type model: :class:`epynn.network.models.EpyNN`

    :param timeout: Time to hold on initialization logs.
    :type timeout: int
    """
    model.init_logs = []

    # Dataset initialization logs
    dsets = model.embedding.dsets
    se_dataset = model.embedding.se_dataset

    model.init_logs.append(dsets_samples_logs(dsets, se_dataset))
    model.init_logs.append(dsets_labels_logs(dsets))

    # Model architecture and shapes initialization logs
    network = model.network

    model.init_logs.append(network_logs(network))

    # Model and layer hyperparameters initialization logs
    layers = model.layers

    model.init_logs.append(layers_lrate_logs(layers))
    model.init_logs.append(layers_others_logs(layers))

    initialize_logs_print(model)

    start_counter(timeout)

    return None
