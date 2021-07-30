# EpyNN/nnlibs/commons/logs.py
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


def model_logs(model):
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

    logs_freq = model.se_config['logs_frequency']
    logs_freq_disp = model.se_config['logs_frequency_display']

    if model.e == 1:
        model.current_logs = [headers_logs(model.metrics, colors)]

    if (model.e + 1) % logs_freq == 0:
        model.current_logs.append(current_logs(model, colors))

    if len(model.current_logs) == logs_freq_disp + 1 or model.e == model.epochs - 1:

        logs = tabulate(model.current_logs,
                        headers="firstrow",
                        numalign="center",
                        stralign='center',
                        tablefmt="pretty",
                        )

        print (logs, flush=True)

        remaining_time_logs(model)

        model.current_logs = [headers_logs(model.metrics, colors)]

    return None


def current_logs(model, colors):
    """.

    :param model:
    :type model:

    :param colors:
    :type colors:

    :return:
    :rtype:
    """
    metrics = model.metrics
    dsets = model.embedding.dsets

    log = []

    log.append(colored(model.e, 'white', attrs=['bold']))

    for layer in model.layers:
        log.append(colored("{:.2e}".format(layer.lrate[model.e]), 'white', attrs=['bold']))

    for i, s in enumerate(metrics.keys()):

        i = (i+1) % len(colors)

        for k in range(len(dsets)):

            m = round(metrics[s][k][-1],3)

            log.append(colored('%.3f' % m, colors[i], attrs=['bold']))

    if model.saved == True:
        log.append(colored('SAVED','red','on_white',attrs=['bold','blink']))
        model.saved = False
    else:
        log.append(colored(model.uname,'white',attrs=[]))

    return log


def remaining_time_logs(model):
    """.

    :param model:
    :type model:
    """
    elapsed_time = round(time.time() - int(model.ts), 2)

    rate = round((model.e+1) / elapsed_time, 2)

    ttc = round((model.epochs - model.e + 1) / rate)

    cprint ('TIME: %ss RATE: %se/s TTC: %ss' % (elapsed_time, rate, ttc), 'white', attrs=['bold'])

    return None


def initialize_model_logs(model):
    """.

    :param model:
    :type model:
    """
    model.current_logs = []

    model.init_logs = []

    #
    dsets = model.embedding.dsets
    se_dataset = model.se_dataset
    se_config = model.se_config

    model.init_logs.append(dsets_samples_logs(dsets, se_dataset, se_config))
    model.init_logs.append(dsets_labels_logs(dsets))

    #
    network = model.network

    model.init_logs.append(network_logs(network))

    #
    layers = model.layers

    model.init_logs.append(layers_lrate_logs(layers))
    model.init_logs.append(layers_others_logs(layers))

    initialize_logs_print(model)

    return None


def initialize_logs_print(model):
    """.

    :param model:
    :type model:
    """
    cprint ('----------------------- %s -------------------------\n' % model.name, attrs=['bold'], end='\n\n')

    cprint ('-------------------------------- Datasets ------------------------------------\n',attrs=['bold'])

    print (model.init_logs[0].draw(), end='\n\n')
    print (model.init_logs[1].draw(), end='\n\n')


    cprint ('----------------------- Model Architecture -------------------------\n',attrs=['bold'])

    print (model.init_logs[2].draw(), end='\n\n')


    cprint ('------------------------------------------- Layers ---------------------------------------------\n',attrs=['bold'])

    print (model.init_logs[3].draw(), end='\n\n')
    print (model.init_logs[4].draw(), end='\n\n')


    cprint ('----------------------- %s -------------------------\n' % model.name, attrs=['bold'], end='\n\n')

    return None


def network_logs(network):
    """.

    :param network:
    :type network:

    :return:
    :rtype:
    """
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

    for i, layer in enumerate(network.values()):

        log = []

        log.append(str(i))
        log.append(layer['Layer'])

        for key in ['Dimensions', 'Activation', 'FW_Shapes', 'BW_Shapes']:
            log.append('\n'.join([k + ': ' + str(v) for k,v in layer[key].items()]))

        logs.add_row(log)

    logs.set_max_width(0)

    return logs


def layers_lrate_logs(layers):
    """.

    :param layers:
    :type layers:

    :return:
    :rtype:
    """
    headers = [
        'Layer',
        'training\nepochs\n(e)',
        'schedule\nmode',
        'cycling\n(n)',
        'e/n',
        'decay\n(k)',
        'descent\n(d)',
        'start',
        'end',
        'end\n(%)',
    ]

    logs = Texttable()

    logs.add_row(headers)

    for layer in layers:

        log = []

        se_hPars = layer.se_hPars

        lr_ori = layer.lrate[0]
        lr_end = layer.lrate[-1]

        pc_end = round(lr_end / lr_ori * 100,3)

        log.append(layer.name)
        log.append(se_hPars['training_epochs'])
        log.append(se_hPars['schedule_mode'])
        log.append(se_hPars['cycling_n'])
        log.append(se_hPars['epochs_per_cycle'])
        log.append(se_hPars['decay_k'])
        log.append(se_hPars['descent_d'])
        log.append("{:.2e}".format(lr_ori))
        log.append("{:.2e}".format(lr_end))
        log.append(pc_end)

        logs.add_row(log)

    logs.set_max_width(0)

    return logs


def layers_others_logs(layers):
    """.

    :param layers:
    :type layers:

    :return:
    :rtype:
    """
    headers = [
        'Layer',
        'LRELU\nalpha',
        'ELU\nalpha',
        'softmax\ntemperature',
        'min.\nepsilon',
        'reg.\nl1',
        'reg.\nl2',
    ]

    logs = Texttable()

    logs.add_row(headers)

    for layer in layers:

        se_hPars = layer.se_hPars

        log = []

        log.append(layer.name)
        log.append(se_hPars['LRELU_alpha'])
        log.append(se_hPars['ELU_alpha'])
        log.append(se_hPars['softmax_temperature'])
        log.append(se_hPars['min_epsilon'])
        log.append(se_hPars['regularization_l1'])
        log.append(se_hPars['regularization_l2'])

        logs.add_row(log)

    logs.set_max_width(0)

    return logs


def dsets_samples_logs(dsets, se_dataset, se_config):
    """.

    :param dsets:
    :type dsets:

    :param se_dataset:
    :type se_dataset:

    :param se_config:
    :type se_config:

    :return:
    :rtype:
    """
    headers = [
        'N_SAMPLES',
        'dtrain\n(0)',
        'dtest\n(1)',
        'dval\n(2)',
        'batch\nnumber\n(b)',
        'dtrain/b',
        'dataset\ntarget',
        'metrics\ntarget',
    ]

    logs = Texttable()

    logs.add_row(headers)

    batch_number = se_dataset['batch_number']

    sample_per_batch = int(dsets[0].s) // int(batch_number)

    log = []

    log.append(se_dataset['N_SAMPLES'])
    log.append(dsets[0].s)
    log.append(dsets[1].s)
    log.append(dsets[2].s)
    log.append(batch_number)
    log.append(sample_per_batch)
    log.append(se_config['dataset_target'])
    log.append(se_config['metrics_target'])

    logs.add_row(log)

    logs.set_max_width(0)

    return logs


def dsets_labels_logs(dsets):
    """.

    :return:
    :rtype:
    """
    headers = [
        'N_LABELS',
        'dtrain\n(0)',
        'dtest\n(1)',
        'dval\n(2)',
    ]

    logs = Texttable()

    logs.add_row(headers)

    log = []

    log.append(len(dsets[0].b.keys()))

    for dset in dsets:
        log.append('\n'.join([str(k) + ': ' + str(v) for k,v in sorted(dset.b.items())]))

    logs.add_row(log)

    return logs


def headers_logs(metrics,colors):
    """.

    :return:
    :rtype:
    """
    headers = [
        colored('epoch', 'white', attrs=['bold']) + '\n',
    ]

    for i, s in enumerate(['lr']+list(metrics.keys())):

        i = i % len(colors)

        headers.append('\n' + colored('(0)', colors[i], attrs=['bold']))

        headers.append(
            colored('%s' % s, colors[i], attrs=['bold'])
            + '\n'
            + colored('(1)', colors[i], attrs=['bold'])
        )

        headers.append('\n' + colored('(2)', colors[i], attrs=['bold']))

    headers.append(colored('Experiment', 'white', attrs=[]) + '\n')

    return headers


def set_highlighted_excepthook():
    """.
    """
    lexer = get_lexer_by_name('pytb' if sys.version_info.major < 3 else 'py3tb')

    formatter = TerminalTrueColorFormatter(bg='dark', style='fruity')

    def myexcepthook(type, value, tb):
        tbtext = ''.join(traceback.format_exception(type, value, tb))
        sys.stderr.write(highlight(tbtext, lexer, formatter))

        return None

    sys.excepthook = myexcepthook

    return None


def process_logs(msg, level=0):
    """.

    :param msg:
    :type msg:

    :param level:
    :type level:
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


def pretty_json(data):
    """.

    :param data:
    :type data:

    :return:
    :rtype:
    """
    data = json.dumps(data, sort_keys=False, indent=4)

    data = highlight(data, lexers.JsonLexer(), formatters.TerminalFormatter())

    return data
