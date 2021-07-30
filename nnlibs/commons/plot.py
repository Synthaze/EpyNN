# EpyNN/nnlibs/commons/plot.py
# Related third party imports
import termplotlib
from matplotlib import pyplot as plt


def pyplot_metrics(model):
    """.
    """
    se_config = model.se_config

    set_names = [
        'Training',
        'Testing',
        'Validation',
    ]

    plt.figure()

    for s in model.se_config['metrics_plot']:

        for k, dname in enumerate(set_names):

            x = [x for x in range(len(model.metrics[s][k]))]

            y = model.metrics[s][k]

            plt.plot(x, y, label=dname + ' ' + s)

    # x = range(len(hPars.l))
    # y = [ x / max(hPars.l) for x in hPars.l ]
    #
    # plt.plot(x,y,label='lr (Norm.)')

    plt.legend()

    plt.xlabel('Epoch')
    plt.ylabel('Value')

    plt.title(model.uname)

    if se_config['plot_display'] == True:
        plt.show()

    if se_config['plot_save'] == True:
        pass
        #plt.savefig()

    return None


def gnuplot_accuracy(model):
    """.
    """
    se_config = model.se_config

    set_names = [
        'Training',
        'Testing',
        'Validation',
    ]

    fig = termplotlib.figure()

    s = 'accuracy'

    for k, dname in enumerate(set_names):

        x = [x for x in range(len(model.metrics[s][k]))]

        y = model.metrics[s][k]

        fig.plot(x, y, label=dname + ' ' + s, width=50, height=15)

    if se_config['plot_display'] == True:
        fig.show()

    return None
