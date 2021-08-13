# EpyNN/nnlibs/network/report.py
# Standard library imports
import time

# Related third party imports
from tabulate import tabulate
from termcolor import cprint

# Local application/library specific imports
from nnlibs.network.evaluate import batch_evaluate
from nnlibs.commons.logs import (
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
    """.

    :param model:
    :type model:
    """
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

    logs_freq_disp = 11

    if not hasattr(model, 'current_logs'):
        model.current_logs = []

    if model.e == 0:
        model.current_logs = [headers_logs(model, colors)]

    if model.e % model.verbose == 0 or model.e == model.epochs - 1:
        model.current_logs.append(current_logs(model, colors))

    if len(model.current_logs) == logs_freq_disp + 1 or model.e == model.epochs - 1:

        logs = tabulate(model.current_logs,
                        headers="firstrow",
                        numalign="center",
                        stralign='center',
                        tablefmt="pretty",
                        )

        print (logs, flush=True)

        model.current_logs = [headers_logs(model, colors)]

    return None


def initialize_model_report(model, timeout):
    """.

    :param model:
    :type model:
    """
    model.current_logs = []

    model.init_logs = []

    #
    dsets = model.embedding.dsets
    se_dataset = model.embedding.se_dataset

    model.init_logs.append(dsets_samples_logs(dsets, se_dataset))
    model.init_logs.append(dsets_labels_logs(dsets))

    #
    network = model.network

    model.init_logs.append(network_logs(network))

    #
    layers = model.layers

    model.init_logs.append(layers_lrate_logs(layers))
    model.init_logs.append(layers_others_logs(layers))

    initialize_logs_print(model)

    start_counter(timeout)

    return None


def single_batch_report(model, batch, A):
    """.

    :param model:
    :type model:
    """
    model.cts = time.time()

    elapsed_time = round(int(model.cts) - int(model.ts), 2)

    rate = round((model.e + 1) / (elapsed_time + 1e-10), 2)

    ttc = round((model.epochs - model.e + 1) / rate)

    accuracy, cost = batch_evaluate(model, batch.Y, A)

    accuracy = round(accuracy, 3)
    cost = round(cost, 5)

    batch_counter = batch.name + '/' + model.embedding.batch_dtrain[-1].name

    cprint('Epoch %s - Batch %s - Accuracy: %s Cost: %s - TIME: %ss RATE: %se/s TTC: %ss' % (model.e, batch_counter, accuracy, cost, elapsed_time, rate, ttc), 'white', attrs=['bold'], end='\r')

    return None
