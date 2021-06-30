#EpyNN/nnlibs/commons/logs.py
from pygments.formatters import TerminalTrueColorFormatter
from pygments.lexers import get_lexer_by_name
from termcolor import cprint, colored
from texttable import Texttable
from pygments import highlight
from tabulate import tabulate
import traceback
import sys


#@log_function
def model_core_logs(model,dsets,hPars,runData):

    colors = ['green','red','magenta','cyan','yellow','blue','grey']

    if runData.b['i'] == True:

        init_core_logs(model,dsets,hPars,runData,colors)

        runData.print = runData.logs[0].copy()

    logs = []

    logs.append(colored(hPars.e,'white',attrs=['bold']))

    logs.append(colored("{:.2e}".format(hPars.l[hPars.e]),'white',attrs=['bold']))

    for i, s in enumerate(runData.s.keys()):

        i = i % len(colors)

        for k in range(len(dsets)):

            log = round(runData.s[s][k][-1],3)

            logs.append(colored('%.3f' % log,colors[i],attrs=['bold']))

    if runData.b['s'] == True:

        logs.append(colored('SAVED','red','on_white',attrs=['bold','blink']))

        runData.b['s'] = False

    else:

        logs.append(colored(runData.m['nt'],'white',attrs=[]))

    runData.print.extend([logs])

    if len(runData.print) == runData.m['l'] + 1 or hPars.e == hPars.i - 1:

        print (tabulate(runData.print,headers="firstrow",numalign="center",stralign='center',tablefmt="pretty"),flush=True)

        runData.print = runData.logs[0].copy()

    return None


def init_core_logs(model,dsets,hPars,runData,colors):

    print ('\n')

    init_1 = log_model_network(model)

    init_2 = log_lr_schedule(hPars)

    init_3 = log_datasets(dsets,hPars,runData)

    init_4 = log_others(dsets,hPars,runData)

    headers = headers_log(runData,colors)

    runData.logs = [headers,init_1,init_2,init_3,init_4]

    runData.b['i'] = False

    return None


def headers_log(runData,colors):

    headers = [colored('epoch','white',attrs=['bold'])+'\n',colored('lr','white',attrs=['bold'])+'\n']

    for i, s in enumerate(runData.s.keys()):

        i = i % len(colors)

        headers.append('\n'+colored('(0)',colors[i],attrs=['bold']))

        headers.append(colored('%s' % s,colors[i],attrs=['bold'])+'\n'+colored('(1)',colors[i],attrs=['bold']))

        headers.append('\n'+colored('(2)',colors[i],attrs=['bold']))

    headers.append(colored('Experiment','white',attrs=[])+'\n')

    return [headers]


def log_model_network(model):

    model.logs()

    colors = {
        'Flatten': 'grey',
        'Dense': 'red',
        'LSTM': 'cyan',
        'RNN': 'blue',
        'GRU': 'magenta',
        'Convolution': 'green',
        'Dropout': 'yellow',
        'Pooling': 'green'
        }

    headers = ['ID','Layer','Dimensions','Activation','Shapes']

    logs = Texttable()

    logs.add_rows([headers])

    for i, layer in enumerate(model.a):

        log = [str(i),layer['Layer'],'\n'.join(layer['Dimensions']),'\n'.join(layer['Activation']),'\n'.join(layer['Shapes'])]

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


def log_datasets(dsets,hPars,runData):

    headers = ['N_SAMPLES','dtrain\n(0)','dtest\n(1)','dval\n(2)','batch\nnumber\n(b)','dtrain/b','dataset\ntarget','metrics\ntarget']

    logs = Texttable()

    logs.add_rows([headers])

    log = [runData.m['s'],dsets[0].s,dsets[1].s,dsets[2].s,hPars.b,int(dsets[0].s)//int(hPars.b),runData.m['d'],runData.m['m']]

    logs.add_row(log)

    logs.set_max_width(0)

    cprint ('-------------------------------- Datasets ------------------------------------\n',attrs=['bold'])

    print (logs.draw())

    print ('\n')

    return logs


def log_others(dsets,hPars,runData):

    headers = ['reg.\nl1','reg.\nl2','min.\nepsilon','LRELU\nalpha','ELU\nalpha','softmax\ntemperature']

    logs = Texttable()

    logs.add_rows([headers])

    log = [hPars.c['l1'],hPars.c['l2'],hPars.c['E'],hPars.c['e'],hPars.c['l'],hPars.c['s']]

    logs.add_row(log)

    logs.set_max_width(0)

    cprint ('-------------------------- Others ---------------------\n',attrs=['bold'])

    print (logs.draw())

    print ('\n')

    headers = ['model\nsave','dsets\nsave','hPars\nsave','runData\nsave','plot\nsave']

    logs = Texttable()

    logs.add_rows([headers])

    log = [str(runData.b['ms']),str(runData.b['ds']),str(runData.b['hs']),str(runData.b['rs']),str(runData.b['ps'])]

    logs.add_row(log)

    logs.set_max_width(0)

    cprint ('-------------------- Save -----------------\n',attrs=['bold'])

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
