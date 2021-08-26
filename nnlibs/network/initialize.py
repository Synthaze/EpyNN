# EpyNN/nnlibs/network/initialize.py
# Standard library imports
import sys

# Related third party imports
from termcolor import cprint
import numpy as np

# Local application/library specific imports
from nnlibs.commons.logs import pretty_json


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

        layer.check_fw = False
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

        model.network[id(layer)]['FW_Shapes'] = layer.fs

        layer.check_fw = True

    dX = dA = model.training_loss(Y, A, deriv=True)

    for layer in reversed(model.layers):

        layer.check_bw = False

        cprint('Layer: ' + layer.name, attrs=['bold'], end=end)

        cprint('backward: ' + layer.name, 'cyan', attrs=['bold'], end=end)
        dX = layer.backward(dX)

        model.network[id(layer)]['BW_Shapes'] = layer.bs

        cprint('compute_gradients: ' + layer.name, 'cyan', attrs=['bold'], end=end)
        layer.compute_gradients()

        layer.check_bw = True

    cprint('--- EpyNN Check OK! --- ', attrs=['bold'], end=end)

    model.e = 0

    return None


def model_initialize_exceptions(model,trace):
    """Handle error in model initialization and show logs.
    """

    layers_fw = [layer for layer in model.layers if hasattr(layer, 'check_fw')]
    layers_bw = reversed([layer for layer in model.layers if hasattr(layer, 'check_bw')])

    cprint('\n/!\\ Initialization of EpyNN model failed', 'red', attrs=['bold'])

    # for layer in layers_fw:
    #     layer.update_shapes(layer.fc, layer.fs)
    #     if layer.check_fw:
    #         cprint('\n%s FW OK' % layer.name, 'white',  attrs=['bold'])
    #         print('Output shape FW:', layer.fs['A'])
    #     if not layer.check_fw:
    #         cprint('\n%s layer has crashed' % layer.name, 'red',  attrs=['bold'])
    #         cprint('\nDimensions (layer.d):', 'white',  attrs=['bold'])
    #         print(', '.join([k + ': ' + str(v) for k, v in layer.d.items()]))
    #         cprint('\nForward shapes (layer.fs):', 'green',  attrs=['bold'])
    #         print('\n'.join([k + ': ' + str(v) for k, v in layer.fs.items()]))
    #         break
    #
    # for layer in layers_bw:
    #     layer.update_shapes(layer.bc, layer.bs)
    #     if layer.check_bw:
    #         cprint('\n%s BW OK' % layer.name, 'white',  attrs=['bold'])
    #         print('Output shape BW:', layer.bs['dX'])
    #     if not layer.check_bw:
    #         cprint('\n%s BW has crashed' % layer.name, 'red',  attrs=['bold'])
    #         cprint('\nDimensions (layer.d):', 'white',  attrs=['bold'])
    #         print(', '.join([k + ': ' + str(v) for k, v in layer.d.items()]))
    #         cprint('\nForward shapes (layer.fs):', 'green',  attrs=['bold'])
    #         print('\n'.join([k + ': ' + str(v) for k, v in layer.fs.items()]))
    #         cprint('\nBackward shapes (layers.bs):', 'cyan',  attrs=['bold'])
    #         print('\n'.join([k + ': ' + str(v) for k, v in layer.bs.items()]))
    #         break

    print(trace)

    sys.exit()
