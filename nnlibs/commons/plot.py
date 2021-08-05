# EpyNN/nnlibs/commons/plot.py
# Standard library imports
import os

# Related third party imports
import termplotlib
from matplotlib import pyplot as plt

# Local application/library specific imports
from nnlibs.commons.logs import process_logs


def pyplot_metrics(model):
    """Plot metrics from training with matplotlib

    :param model: An instance of EpyNN network.
    :type model: :class:`nnlibs.meta.models.EpyNN`
    """

    plt.figure()

    for s in model.se_config['metrics_plot']:

        for k, dset in enumerate(model.embedding.dsets):

            dname = dset.name

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

    plt.show()

    plot_path = os.path.join(os.getcwd(), 'plots', model.uname)  + '.png'

    plt.savefig(plot_path)

    process_logs('Make: ' + plot_path, level=1)

    return None


def gnuplot_accuracy(model):
    """Plot metrics from training with gnuplot

    :param model: An instance of EpyNN network.
    :type model: :class:`nnlibs.meta.models.EpyNN`
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

        fig.show()

    return None
