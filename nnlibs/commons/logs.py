#EpyNN/nnlibs/commons/logs.py
from pygments.formatters import TerminalTrueColorFormatter
from pygments.lexers import get_lexer_by_name
from pygments import highlight
from termcolor import cprint
import traceback
import sys


#@log_function
def model_core_logs(model,dsets,hPars,runData):

    colors = ['green','red','magenta','cyan','yellow','blue','grey']

    if runData.b['i'] == True:

        init_core_logs(runData,colors)

    cprint(hPars.e,'white',attrs=['bold'],end=' | ')

    cprint(hPars.l[hPars.e],'white',attrs=['bold'],end=' | ')

    for i, s in enumerate(runData.s.keys()):

        i = i % len(colors)

        for k in range(len(dsets)):

            log = round(runData.s[s][k][-1],3)

            cprint('%.3f' % log,colors[i],attrs=['bold'],end=' | ')

    if runData.b['s'] == True:

        cprint('SAVED','red','on_white',attrs=['bold','blink'],end=' ')

        runData.b['s'] = False

    else:

        for k in range(len(dsets)):

            cprint(dsets[k].s,'white',attrs=[],end=' | ')

        cprint(runData.m['nt'],'white',attrs=[],end=' ')

    print (' ',flush=True)

    return None


def init_core_logs(runData,colors):

    cprint('epoch','white',attrs=['bold'],end=' | ')

    cprint('lr','white',attrs=['bold'],end=' | ')

    for i, s in enumerate(runData.s.keys()):

        i = i % len(colors)

        cprint('%s' % s,colors[i],attrs=['bold'],end=' | ')

    cprint('Samples','white',attrs=[],end=' | ')

    cprint('Experiment','white',attrs=[],end=' ')

    print (' ',flush=True)

    runData.b['i'] = False

    return None


def log_lr_schedule(hPars):

    lr_ori = hPars.l[0]
    lr_end = round(hPars.l[-1],9)
    logs = (str(lr_ori),str(lr_end),str(round((1-lr_end/lr_ori)*100,4)))

    cprint ('\n===\nLearning rate - Start: %s - End: %s - Decay %s%%\n===' % logs,'red','on_white',attrs=['bold'],end='\n')


def set_highlighted_excepthook():

    lexer = get_lexer_by_name("pytb" if sys.version_info.major < 3 else "py3tb")

    formatter = TerminalTrueColorFormatter(bg="dark",style="fruity")

    def myexcepthook(type, value, tb):
        tbtext = ''.join(traceback.format_exception(type, value, tb))
        sys.stderr.write(highlight(tbtext, lexer, formatter))

    sys.excepthook = myexcepthook
