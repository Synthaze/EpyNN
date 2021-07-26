#EpyNN/nnlibs/commons/logs.py
from pygments.formatters import TerminalTrueColorFormatter
from pygments import highlight, lexers, formatters
from pygments.lexers import get_lexer_by_name
from termcolor import cprint, colored
from texttable import Texttable
from pygments import highlight
from tabulate import tabulate
import traceback
import time
import sys
import json

#@log_function
def model_logs(model):

    colors = ['green','red','magenta','cyan','yellow','blue','grey']

    embedding = model.layers[0]

    dsets = [embedding.dtrain,embedding.dtest,embedding.dval]

    if model.init == True:


        metrics.print = metrics.logs[0].copy()

    logs = []

    logs.append(colored(hPars.e,'white',attrs=['bold']))

    logs.append(colored("{:.2e}".format(hPars.l[hPars.e]),'white',attrs=['bold']))

    for i, s in enumerate(metrics.s.keys()):

        i = i % len(colors)

        for k in range(len(dsets)):

            log = round(metrics.s[s][k][-1],3)

            logs.append(colored('%.3f' % log,colors[i],attrs=['bold']))

    if metrics.b['s'] == True:

        logs.append(colored('SAVED','red','on_white',attrs=['bold','blink']))

        metrics.b['s'] = False

    else:

        logs.append(colored(metrics['nt'],'white',attrs=[]))

    if hPars.e % metrics['f'] == 0:
        metrics.print.extend([logs])

    if len(metrics.print) == metrics['fd'] + 1 or hPars.e == hPars.i - 1:

        print (tabulate(metrics.print,headers="firstrow",numalign="center",stralign='center',tablefmt="pretty"),flush=True)
        t = round(time.time()-int(metrics['t']),2)

        rate = round( (hPars.e + 1 ) / t,2)

        ttc = round(( hPars.i - hPars.e + 1 ) / rate)

        cprint ('TIME: %ss RATE: %se/s TTC: %ss' % (t,rate,str(ttc)),'white',attrs=['bold'])

        metrics.print = metrics.logs[0].copy()


    if hPars.e + 1 == hPars.i:
        for i in range(1,5):
            print (metrics.logs[i].draw())


    return None


def initialize_model_logs(model):

    embedding = model.layers[0]

    dsets = [embedding.dtrain,embedding.dtest,embedding.dval]

    hPars = model.se_hPars

    metrics = model.metrics

    print ('\n')

    init_1 = log_model_network(model)

    init_2 = log_datasets(model,dsets,hPars,metrics)

    init_3 = [] #log_lr_schedule(hPars)

    init_4 = [] #log_others(dsets,hPars,metrics)

    cprint ('----------------------- %s -------------------------\n' % model.name,attrs=['bold'])

    print ('\n')
    colors = ['green','red','magenta','cyan','yellow','blue','grey']

    headers = headers_log(metrics,colors)

    model.logs = [headers,init_1,init_2,init_3,init_4]

    model.init = False

    return None


def headers_log(metrics,colors):
    colors = ['green','red','magenta','cyan','yellow','blue','grey']

    headers = [colored('epoch','white',attrs=['bold'])+'\n',colored('lr','white',attrs=['bold'])+'\n']

    for i, s in enumerate(metrics.keys()):

        i = i % len(colors)

        headers.append('\n'+colored('(0)',colors[i],attrs=['bold']))

        headers.append(colored('%s' % s,colors[i],attrs=['bold'])+'\n'+colored('(1)',colors[i],attrs=['bold']))

        headers.append('\n'+colored('(2)',colors[i],attrs=['bold']))

    headers.append(colored('Experiment','white',attrs=[])+'\n')

    return [headers]


def log_model_network(model):

    headers = ['ID','Layer','Dimensions','Activation','FW_Shapes','BW_Shapes']

    logs = Texttable()

    logs.add_rows([headers])

    for i, layer in enumerate(model.network.values()):

        log = [str(i),layer['Layer'],'\n'.join(layer['Dimensions']),'\n'.join(layer['Activation']),'\n'.join(layer['FW_Shapes']),'\n'.join(layer['BW_Shapes'])]

        logs.add_row(log)

    logs.set_max_width(0)

    cprint ('----------------------- Model Architecture -------------------------\n',attrs=['bold'])

    print (logs.draw())

    print ('\n')

    return logs


def log_lr_schedule(hPars):


    headers = ['training\nepochs\n(e)','schedule\nmode','cycling\n(n)','e/n','decay\n(k)','descent\n(d)','start','end','end\n(%)']

    logs = Texttable()

    logs.add_rows([headers])

    pc_end = round(hPars.l[-1] / hPars.s['l'] * 100,3)

    lr_ori = "{:.2e}".format(hPars.s['l'])
    lr_end = "{:.2e}".format(hPars.l[-1])

    log = [hPars.i,hPars.s['s'],hPars.s['n'],hPars.s['c'],hPars.s['k'],hPars.s['d'],lr_ori,lr_end,pc_end]

    logs.add_row(log)

    logs.set_max_width(0)

    cprint ('-------------------------------- Learning rate -----------------------------------\n',attrs=['bold'])

    print (logs.draw())

    print ('\n')

    return logs


def log_datasets(model,dsets,hPars,metrics):

    headers = ['N_SAMPLES','dtrain\n(0)','dtest\n(1)','dval\n(2)','batch\nnumber\n(b)','dtrain/b','dataset\ntarget','metrics\ntarget']

    logs = Texttable()

    logs.add_rows([headers])

    batch_number = model.se_dataset['batch_number']
    n_samples = model.se_dataset['N_SAMPLES']
    dataset_target = model.se_config['dataset_target']
    metrics_target = model.se_config['metrics_target']

    log = [n_samples,dsets[0].s,dsets[1].s,dsets[2].s,batch_number,int(dsets[0].s)//int(batch_number),dataset_target,metrics_target]

    logs.add_row(log)

    logs.set_max_width(0)

    cprint ('-------------------------------- Datasets ------------------------------------\n',attrs=['bold'])

    print (logs.draw())

    print ('\n')

    headers = ['N_LABELS','dtrain\n(0)','dtest\n(1)','dval\n(2)']

    logs = Texttable()

    logs.add_rows([headers])


    N_LABELS = len(dsets[0].b.keys())

    dtrain_balance = '\n'.join([ str(k)+': '+ str(v) for k,v in sorted(dsets[0].b.items()) ])
    dtest_balance = '\n'.join([ str(k)+': '+ str(v) for k,v in sorted(dsets[1].b.items()) ])
    dval_balance = '\n'.join([ str(k)+': '+ str(v) for k,v in sorted(dsets[2].b.items()) ])

    log = [N_LABELS,dtrain_balance,dtest_balance,dval_balance]

    logs.add_row(log)

    print (logs.draw())

    print ('\n')

    return logs


def log_others(dsets,hPars,metrics):

    headers = ['reg.\nl1','reg.\nl2','min.\nepsilon','LRELU\nalpha','ELU\nalpha','softmax\ntemperature']

    logs = Texttable()

    logs.add_rows([headers])

    log = [hPars.c['l1'],hPars.c['l2'],hPars.c['E'],hPars.c['l'],hPars.c['e'],hPars.c['s']]

    logs.add_row(log)

    logs.set_max_width(0)

    cprint ('-------------------------- Others ---------------------\n',attrs=['bold'])

    print (logs.draw())

    print ('\n')

    return logs


def set_highlighted_excepthook():

    lexer = get_lexer_by_name("pytb" if sys.version_info.major < 3 else "py3tb")

    formatter = TerminalTrueColorFormatter(bg="dark",style="fruity")

    def myexcepthook(type, value, tb):
        tbtext = ''.join(traceback.format_exception(type, value, tb))
        sys.stderr.write(highlight(tbtext, lexer, formatter))

    sys.excepthook = myexcepthook




def pretty_json(data):
    print (highlight(json.dumps(data,sort_keys=False,indent=4), lexers.JsonLexer(), formatters.TerminalFormatter()))
    return data
