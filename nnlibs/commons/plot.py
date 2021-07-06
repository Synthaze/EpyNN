#EpyNN/nnlibs/commons/plot.py
from nnlibs.commons.decorators import *

import matplotlib.pyplot as plt
import termplotlib as tpl


@log_function
def pyplot_metrics(model,hPars,runData):

    set_names = ['Training','Testing','Validation']

    plt.figure()

    for s in runData.m['p']:

        for k, dset in enumerate(set_names):

            x = [ x for x in range(len(runData.s[s][k]))]
            y = runData.s[s][k]

            plt.plot(x,y,label=dset+' '+s)

    x = range(len(hPars.l))
    y = [ x / max(hPars.l) for x in hPars.l ]

    plt.plot(x,y,label='lr (Norm.)')

    plt.legend()

    plt.xlabel('Epoch'), plt.ylabel('Value')

    t = runData.m['nt']+'\n'+model.n

    plt.title(t)

    if runData.b['pd'] == True:
        plt.show()

    if runData.b['ps'] == True:
        plt.savefig(runData.p['ps'])

    return None


@log_function
def gnuplot_accuracy(runData):

    set_names = ['Training','Testing','Validation']

    fig = tpl.figure()

    s = 'accuracy'

    for k, dset in enumerate(set_names):

        x = [ x for x in range(len(runData.s[s][k]))]
        y = runData.s[s][k]

        fig.plot(x,y, label=dset+' '+s, width=50, height=15)

    if runData.b['pd'] == True:
        fig.show()

    return None
