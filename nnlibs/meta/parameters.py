# EpyNN/nnlibs/meta/parameters.py
# Related third party imports
import numpy as np

# Local application/library specific imports
from nnlibs.commons.schedule import schedulers


def assign_seed_layers(model):
    """Assign seed and independant pseudo-random generators for each layer in model.

    :param model: An instance of EpyNN network.
    :type model: :class:`nnlibs.meta.models.EpyNN`
    """
    seed = model.seed

    for layer in model.layers:

        if seed != None:
            layer.o['seed'] = seed
            seed += 1

        else:
            layer.o['seed'] = None

        layer.np_rng = np.random.default_rng(seed=layer.o['seed'])

    return None


def compute_learning_rate(model):
    """Schedule learning rate for each layer in model.

    :param model: An instance of EpyNN network.
    :type model: :class:`nnlibs.meta.models.EpyNN`
    """
    for layer in model.layers:

        if layer.se_hPars == None:
            se_hPars = layer.se_hPars = model.se_hPars
        else:
            se_hPars = layer.se_hPars

        layer.se_hPars, layer.lrate = schedulers(se_hPars, model.epochs)

    return None
