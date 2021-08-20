# EpyNN/nnlibs/commons/plot.py
# Standard library imports
import os

# Related third party imports
import termplotlib
from matplotlib import pyplot as plt

# Local application/library specific imports
from nnlibs.commons.logs import process_logs


def pyplot_metrics(model, path):
    """Plot metrics/costs from training with matplotlib.

    :param model: An instance of EpyNN network object.
    :type model: :class:`nnlibs.meta.models.EpyNN`

    :param path: Write matplotlib plot.
    :type path: bool or NoneType
    """
    plt.figure()

    # Iterate over metrics/costs
    for s in model.metrics.keys():

        # Iterate over active datasets
        for k, dset in enumerate(model.embedding.dsets):

            dname = dset.name

            x = [x for x in range(len(model.metrics[s][k]))]

            y = model.metrics[s][k]

            plt.plot(x, y, label=dname + ' ' + s)

    plt.legend()

    plt.xlabel('Epoch')
    plt.ylabel('Value')

    plt.title(model.uname)

    plt.show()

    # If path sets to None, set to defaults - Note path can be set to False, which makes it print only
    if path == None:
        path = 'plots'
        plot_path = os.path.join(os.getcwd(), path, model.uname)  + '.png'

    if path:
        plt.savefig(plot_path)
        process_logs('Make: ' + plot_path, level=1)

    plt.close()

    return None


def gnuplot_accuracy(model):
    """Plot metrics from training with gnuplot. Accuracy only.

    :param model: An instance of EpyNN network object.
    :type model: :class:`nnlibs.meta.models.EpyNN`
    """
    fig = termplotlib.figure()

    s = 'accuracy'

    # Iterate over active datasets
    for k, dset in enumerate(model.embedding.dsets):

        dname = dset.name

        x = [x for x in range(len(model.metrics[s][k]))]

        y = model.metrics[s][k]

        fig.plot(x, y, label=dname + ' ' + s, width=50, height=15)

        fig.show()

    return None
