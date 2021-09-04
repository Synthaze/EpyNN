# EpyNN/nnlibs/network/initialize.py
# Standard library imports
import sys

# Related third party imports
from termcolor import cprint
import numpy as np


def model_initialize(model, params=True, end='\n'):
    """Initialize Neural Network.

    :param model: An instance of EpyNN network.
    :type model: :class:`nnlibs.network.models.EpyNN`

    :param params: Layer parameters initialization, defaults to `True`.
    :type params: bool, optional
    """
    model.network = {id(layer):{} for layer in model.layers}

    model.np_rng = np.random.default_rng(seed=model.seed)
    seed = model.seed + 1 if model.seed else None

    model.embedding.training_batches()

    batch_dtrain = model.embedding.batch_dtrain

    sample = batch_dtrain[0]

    A = X = sample.X
    Y = sample.Y

    cprint('--- EpyNN Check --- ', attrs=['bold'], end=end)

    for layer in model.layers:

        layer.check = False
        layer.name = layer.__class__.__name__
        model.network[id(layer)]['Layer'] = layer.name
        model.network[id(layer)]['Activation'] = layer.activation
        model.network[id(layer)]['Dimensions'] = layer.d

        layer.o['seed'] = seed
        layer.np_rng = np.random.default_rng(seed=layer.o['seed'])
        seed = seed + 1 if seed else None

        cprint('Layer: ' + layer.name, attrs=['bold'], end=end)
        cprint('compute_shapes: ' + layer.name, 'green', attrs=['bold'], end=end)

        layer.compute_shapes(A)

        model.network[id(layer)]['FW_Shapes'] = layer.fs

        if params:
            cprint('initialize_parameters: ' + layer.name, 'green', attrs=['bold'], end=end)

            layer.initialize_parameters()

        cprint('forward: ' + layer.name, 'green', attrs=['bold'], end=end)
        A = layer.forward(A)
        print('shape:', layer.fs['A'], end=end)

        model.network[id(layer)]['FW_Shapes'] = layer.fs

        delattr(layer, 'check')

    dX = dA = model.training_loss(Y, A, deriv=True)

    for layer in reversed(model.layers):

        layer.check = False

        cprint('Layer: ' + layer.name, attrs=['bold'], end=end)
        cprint('backward: ' + layer.name, 'cyan', attrs=['bold'], end=end)
        dX = layer.backward(dX)

        print('shape:', layer.bs['dX'], end=end)

        model.network[id(layer)]['BW_Shapes'] = layer.bs
        cprint('compute_gradients: ' + layer.name, 'cyan', attrs=['bold'], end=end)

        layer.compute_gradients()

        delattr(layer, 'check')

    cprint('--- EpyNN Check OK! --- ', attrs=['bold'], end=end)

    model.e = 0

    return None


def model_initialize_exceptions(model,trace):
    """Handle error in model initialization and show logs.
    """

    cprint('\n/!\\ Initialization of EpyNN model failed - debug', 'red', attrs=['bold'])

    try:

        layer = [layer for layer in model.layers if hasattr(layer, 'check')][0]

        layer.update_shapes(layer.fc, layer.fs)
        layer.update_shapes(layer.bc, layer.bs)

        cprint('%s layer: ' % layer.name, 'red', attrs=['bold'])

        cprint('Known dimensions', 'white', attrs=['bold'])
        print(', '.join([k + ': ' + str(v) for k, v in layer.d.items()]))

        cprint('Known forward shapes', 'green', attrs=['bold'])
        print('\n'.join([k + ': ' + str(v) for k, v in layer.fs.items()]))

        cprint('Known backward shape', 'cyan', attrs=['bold'])
        print('\n'.join([k + ': ' + str(v) for k, v in layer.bs.items()]))

    except:
        pass

    cprint('System trace', 'red', attrs=['bold'])

    print(trace)

    sys.exit()
