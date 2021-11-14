# EpyNN/epynn/commons/logs.py
# Standard library imports
import traceback
import json
import time
import sys
 
# Related third party imports
from pygments import highlight, lexers, formatters
from pygments.formatters import TerminalTrueColorFormatter
from pygments.lexers import get_lexer_by_name
from termcolor import cprint, colored
from texttable import Texttable
from pygments import highlight
from tabulate import tabulate
import numpy as np


def headers_logs(model, colors):
    """Generate headers to log epochs, learning rates, training metrics, costs and experiment name.

    :param model: An instance of EpyNN network object.
    :type model: :class:`epynn.network.models.EpyNN`

    :param colors: Colors that may be used to print logs.
    :type colors: list[str]

    :return: Headers with respect to training setup.
    :rtype: list[str]
    """
    metrics = model.metrics                   # Contains metrics + cost computed along training epochs
    len_dsets = len(model.embedding.dsets)    # Number of active datasets (dtrain, dtest, dval or dtrain, dtest or dtrain)

    # Initialize headers list with first field: epochs
    headers = [
        colored('epoch', 'white', attrs=['bold']) + '\n',
    ]

    for layer in model.layers:

        if layer.p != {}:
            # If layer is trainable, then add header for learning rate.
            headers.append(
                colored('lrate', colors[0], attrs=['bold'])
                + '\n'
                + colored(layer.name, colors[0], attrs=[])
            )

    # Iterate over metrics/costs - We need to adapt headers with respect to the number of active datasets
    for i, s in enumerate(list(metrics.keys())):

        i = (i+1) % len(colors)

        # There is at least one non-empty training set
        headers.append(
            colored('%s' % s, colors[i], attrs=['bold'])
            + '\n'
            + colored('dtrain', colors[i], attrs=['bold'])
        )

        # There may be one non-empty validation set
        if len_dsets == 2:
            headers.append('\n' + colored('dval', colors[i], attrs=['bold']))

        # There may be one non-empty validation and testing set
        if len_dsets == 3:
            headers.append('\n' + colored('dtest', colors[i], attrs=['bold']))

    headers.append(colored('Experiment', 'white', attrs=[]) + '\n')

    return headers


def current_logs(model, colors):
    """Build logs with respect to headers for current epoch which includes epoch, learning rates, training metrics, costs and experiment name.

    :param model: An instance of EpyNN network object.
    :type model: :class:`epynn.network.models.EpyNN`

    :param colors: Colors that may be used to print logs.
    :type colors: list[str]

    :return: Logs for current epoch.
    :rtype: list[str]
    """
    metrics = model.metrics          # Contains metrics + cost computed along training epochs
    dsets = model.embedding.dsets    # Active datasets (dtrain, dval, dtest or dtrain, dval or dtrain)

    log = []

    log.append(colored(model.e, 'white', attrs=['bold']))    # Current epoch

    for layer in model.layers:
        # If layer is trainable, append current learning rate
        if layer.p != {}:
            log.append(colored("{:.2e}".format(layer.lrate[model.e]), 'white', attrs=['bold']))

    # Iterate over metrics/costs
    for i, s in enumerate(metrics.keys()):

        i = (i+1) % len(colors)

        # Iterate over each active dataset
        for k in range(len(dsets)):

            m = round(metrics[s][k][-1], 3)

            log.append(colored('%.3f' % m, colors[i], attrs=['bold']))

    log.append(colored(model.uname,'white',attrs=[]))

    return log


def initialize_logs_print(model):
    """Print model initialization logs which include information about datasets, model architecture and shapes as well as layers hyperparameters.

    :param model: An instance of EpyNN network object.
    :type model: :class:`epynn.network.models.EpyNN`
    """
    cprint ('----------------------- %s -------------------------\n' % model.uname, attrs=['bold'], end='\n\n')


    cprint ('-------------------------------- Datasets ------------------------------------\n',attrs=['bold'])

    print (model.init_logs[0].draw(), end='\n\n')
    print (model.init_logs[1].draw(), end='\n\n')


    cprint ('----------------------- Model Architecture -------------------------\n',attrs=['bold'])

    print (model.init_logs[2].draw(), end='\n\n')


    cprint ('------------------------------------------- Layers ---------------------------------------------\n',attrs=['bold'])

    print (model.init_logs[3].draw(), end='\n\n')
    print (model.init_logs[4].draw(), end='\n\n')


    cprint ('----------------------- %s -------------------------\n' % model.uname, attrs=['bold'], end='\n\n')

    return None


def network_logs(network):
    """Build tabular logs of current network architecture and shapes.

    :param network: Attribute of an instance of EpyNN network object.
    :type network: dict[str, dict[str, str or tuple[int]]]

    :return: Logs for network architecture and shapes.
    :rtype: :class:`texttable.Texttable`
    """
    # List of documented network features
    headers = [
        'ID',
        'Layer',
        'Dimensions',
        'Activation',
        'FW_Shapes',
        'BW_Shapes',
    ]

    logs = Texttable()

    logs.add_row(headers)

    # Iterate over values (layers) in network dictionnary
    for i, layer in enumerate(network.values()):

        log = []

        log.append(str(i))            # Layer index
        log.append(layer['Layer'])    # Layer name

        for key in ['Dimensions', 'Activation', 'FW_Shapes', 'BW_Shapes']:
            log.append('\n'.join([k + ': ' + str(v) for k,v in layer[key].items()]))

        logs.add_row(log)

    logs.set_max_width(0)

    return logs


def layers_lrate_logs(layers):
    """Build tabular logs for layers hyperparameters related to learning rate.

    :param layers: Attribute of an instance of EpyNN network object.
    :type layers: list[Object]

    :return: Logs for layers hyperparameters related to learning rate.
    :rtype: :class:`texttable.Texttable`
    """
    # List of documented layers hyperparameters related to learning rate
    headers = [
        'Layer',
        'epochs',
        'schedule',
        'decay_k',
        'cycle\nepochs\n',
        'cycle\ndescent',
        'cycle\nnumber',
        'learning\nrate\n(start)',
        'learning\nrate\n(end)',
        'end\n(%)',
    ]

    logs = Texttable()

    logs.add_row(headers)

    for layer in layers:

        log = []

        se_hPars = layer.se_hPars    # Local hyperparameters for layer

        lr_ori = layer.lrate[0]      # Learning rate epoch 0
        lr_end = layer.lrate[-1]     # Learning rate last epoch

        pc_end = round(lr_end / lr_ori * 100, 3) if lr_ori != 0 else 0

        log.append(layer.name)
        log.append(se_hPars['epochs'])
        log.append(se_hPars['schedule'])
        log.append(se_hPars['decay_k'])
        log.append(se_hPars['cycle_epochs'])
        log.append(se_hPars['cycle_descent'])
        log.append(se_hPars['cycle_number'])
        log.append("{:.2e}".format(lr_ori))
        log.append("{:.2e}".format(lr_end))
        log.append(pc_end)

        # Document only if layer is trainable
        if layer.p != {}:
            logs.add_row(log)

    logs.set_max_width(0)

    return logs


def layers_others_logs(layers):
    """Build tabular logs for layers hyperparameters related to activation functions.

    :param layers: Attribute of an instance of EpyNN network object.
    :type layers: list[Object]

    :return: Logs for layers hyperparameters related to activation functions.
    :rtype: :class:`texttable.Texttable`
    """
    # List of documented layers hyperparameters related to activation functions
    headers = [
        'Layer',
        'LRELU\nalpha',
        'ELU\nalpha',
        'softmax\ntemperature',
    ]

    logs = Texttable()

    logs.add_row(headers)

    for layer in layers:

        se_hPars = layer.se_hPars    # Local hyperparameters for layer

        log = []

        log.append(layer.name)
        log.append(se_hPars['LRELU_alpha'])
        log.append(se_hPars['ELU_alpha'])
        log.append(se_hPars['softmax_temperature'])

        # Document only if layer is trainable
        if layer.p != {}:
            logs.add_row(log)

    logs.set_max_width(0)

    return logs


def dsets_samples_logs(dsets, se_dataset):
    """Build tabular logs describing datasets.

    :param dsets: Attribute of an instance of embedding layer object. Contains active (non-empty) sets.
    :type dsets: list[:class:`epynn.commons.models.dataSet`]

    :param se_dataset: Attribute of an instance of embedding layer object.
    :type se_dataset: dict[str, int or bool]

    :return: Logs describing datasets.
    :rtype: :class:`texttable.Texttable`
    """
    # List of dataset descriptors.
    headers = [
        'dtrain\n',
        'dval\n',
        'dtest\n',
        'batch\nsize',
    ]

    logs = Texttable()

    logs.add_row(headers)

    batch_size = se_dataset['batch_size']

    log = []

    # Active datasets
    for dset in dsets:
        log.append(len(dset.ids))

    # Fill for inactive datasets
    for i in range(3 - len(dsets)):
        log.append('None')

    log.append(batch_size)

    logs.add_row(log)

    logs.set_max_width(0)

    return logs


def dsets_labels_logs(dsets):
    """Build tabular logs describing datasets Y dimension.

    :param dsets: Attribute of an instance of embedding layer object. Contains active (non-empty) sets.
    :type dsets: list[:class:`epynn.commons.models.dataSet`]

    :return: Logs describing datasets Y dimension.
    :rtype: :class:`texttable.Texttable`
    """
    # List of dataset descriptors for Y dimension.
    headers = [
        'N_LABELS',
        'dtrain\n',
        'dval\n',
        'dtest\n',
    ]

    logs = Texttable()

    logs.add_row(headers)

    log = []

    log.append(len(dsets[0].b.keys()))    # Number of classes

    # Active datasets
    for dset in dsets:
        log.append('\n'.join([str(k) + ': ' + str(v) for k,v in sorted(dset.b.items())]))

    # Fill for inactive datasets
    for i in range(3 - len(dsets)):
        log.append('None')

    logs.add_row(log)

    return logs


def set_highlighted_excepthook():
    """Lexer to pretty print tracebacks.
    """
    # Get lexer from pigmentize/minted
    lexer = get_lexer_by_name('pytb' if sys.version_info.major < 3 else 'py3tb')

    # Colorscheme
    formatter = TerminalTrueColorFormatter(bg='dark', style='fruity')

    # Callback function
    def myexcepthook(type, value, tb):
        tbtext = ''.join(traceback.format_exception(type, value, tb))
        sys.stderr.write(highlight(tbtext, lexer, formatter))

        return None

    sys.excepthook = myexcepthook    # This erase the standard excepthook with the callback

    return None


def process_logs(msg, level=0):
    """Pretty print of EpyNN events.

    :param msg: Message to print on terminal.
    :type msg: str

    :param level: Set color for print, defaults to 0 which renders white.
    :type level: int, optional
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

    cprint(msg, colors[level],  attrs=['bold'])

    return None


def start_counter(timeout=3):
    """Timeout between print of initialization logs and beginning of run.

    :param timeout: Time in seconds, defaults to 3.
    :type timeout: int, optional
    """
    for i in reversed(range(timeout + 1)):
        cprint('Start in %ss' % str(i), attrs=['bold'], end='\r')
        time.sleep(1)

    return None


def pretty_json(network):
    """Pretty json print for traceback during model initialization.

    :param network: Attribute of an instance of EpyNN network object.
    :type network: dict[str, dict[str, str or tuple[int]]]

    :return: Formatted input.
    :rtype: json
    """
    # Convert dict to json
    network = json.dumps(network, sort_keys=False, indent=4)

    # Format for pretty print
    network = highlight(network, lexers.JsonLexer(), formatters.TerminalFormatter())

    return network
