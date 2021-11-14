# EpyNN/epynn/commons/plot.py
# Standard library imports
import os

# Related third party imports
from matplotlib import pyplot as plt

# Local application/library specific imports
from epynn.commons.logs import process_logs
 

def pyplot_metrics(model, path):
    """Plot metrics/costs from training with matplotlib.

    :param model: An instance of EpyNN network object.
    :type model: :class:`epynn.meta.models.EpyNN`

    :param path: Write matplotlib plot.
    :type path: bool or NoneType
    """
    plt.figure()

    metrics = model.metrics    # Contains metrics and cost

    # Iterate over metrics/costs
    for s in metrics.keys():

        # Iterate over active datasets
        for k, dset in enumerate(model.embedding.dsets):

            dname = dset.name

            x = [x for x in range(len(metrics[s][k]))]    # X range

            y = metrics[s][k]    # Y values from metrics[idx_dataset][idx_metrics]

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
