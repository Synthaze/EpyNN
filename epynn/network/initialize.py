# EpyNN/epynn/network/initialize.py
# Standard library imports
import sys

# Related third party imports
from termcolor import cprint
import numpy as np


def model_initialize(model, params=True, end='\n'):
    """Initialize EpyNN network.

    :param model: An instance of EpyNN network.
    :type model: :class:`epynn.network.models.EpyNN`

    :param params: Layer parameters initialization, defaults to `True`.
    :type params: bool, optional

    :param end: Wether to print every line for steps or overwrite, default to `\\n`.
    :type end: str in ['\\n', '\\r']

    :raises Exception: If any layer other than Dense was provided with softmax activation. See :func:`epynn.maths.softmax`.
    """
    # Retrieve sample batch
    model.embedding.training_batches(init=True)
    batch_dtrain = model.embedding.batch_dtrain
    batch = batch_dtrain[0]
    A = batch.X  # Features
    Y = batch.Y  # Labels

    cprint('{: <100}'.format('--- EpyNN Check --- '), attrs=['bold'], end=end)

    # Iterate over layers
    for layer in model.layers:

        # Layer instance attributes
        layer.check = False
        layer.name = layer.__class__.__name__

        cprint('Layer: ' + layer.name, attrs=['bold'], end=end)

        # Store layer information in model summary
        model.network[id(layer)]['Layer'] = layer.name
        model.network[id(layer)]['Activation'] = layer.activation
        model.network[id(layer)]['Dimensions'] = layer.d

        # Dense uses epynn.maths.hadamard to handle softmax derivative
        if 'softmax' in layer.activation.values() and layer.name != 'Dense':
            raise Exception('Softmax can not be used with %s, only with Dense' % layer.name)

        # Test layer.compute_shapes() method
        cprint('compute_shapes: ' + layer.name, 'green', attrs=['bold'], end=end)
        layer.compute_shapes(A)

        # Store forward shapes in model summary
        model.network[id(layer)]['FW_Shapes'] = layer.fs

        # Initialize trainable parameters
        if params:
            cprint('initialize_parameters: ' + layer.name, 'green', attrs=['bold'], end=end)
            layer.initialize_parameters()

        # Test layer.forward() method
        cprint('forward: ' + layer.name, 'green', attrs=['bold'], end=end)
        A = layer.forward(A)

        # Output shape
        print('shape:', layer.fs['A'], end=end)

        # Store updated forward shapes in model summary
        model.network[id(layer)]['FW_Shapes'] = layer.fs

        # Clear check
        delattr(layer, 'check')

    # Compute derivative of loss function
    dX = dA = model.training_loss(Y, A, deriv=True)

    # Iterate over reversed layers
    for layer in reversed(model.layers):

        # Set check attribute for layer
        layer.check = False

        cprint('Layer: ' + layer.name, attrs=['bold'], end=end)

        # Test layer.backward() method
        cprint('backward: ' + layer.name, 'cyan', attrs=['bold'], end=end)
        dX = layer.backward(dX)

        # Output shape
        print('shape:', layer.bs['dX'], end=end)

        # Store backward shapes in model summary
        model.network[id(layer)]['BW_Shapes'] = layer.bs

        # Test layer.compute_gradients() method
        cprint('compute_gradients: ' + layer.name, 'cyan', attrs=['bold'], end=end)
        layer.compute_gradients()

        # Clear check
        delattr(layer, 'check')

    cprint('{: <100}'.format('--- EpyNN Check OK! --- '), attrs=['bold'], end=end)

    # Initialize current epoch to zero
    model.e = 0

    return None


def model_assign_seeds(model):
    """Seed model and layers with independant pseudo-random number generators.

    Model is seeded from user-input. Layers are seeded by incrementing the
    input by one in order to not generate same numbers for all objects

    :param model: An instance of EpyNN network.
    :type model: :class:`epynn.network.models.EpyNN`
    """
    seed = model.seed

    # If seed is not defined, seeding is random
    model.np_rng = np.random.default_rng(seed=seed)

    # Iterate over layers
    for layer in model.layers:

        # If seed is defined
        if seed:
            # We do not want the same seed for every object
            seed += 1

        # Seed layer
        layer.o['seed'] = seed
        layer.np_rng = np.random.default_rng(seed=layer.o['seed'])

    return None


def model_initialize_exceptions(model, trace):
    """Handle error in model initialization and show logs.

    :param model: An instance of EpyNN network.
    :type model: :class:`epynn.network.models.EpyNN`

    :param trace: Traceback of fatal error.
    :type trace: traceback object
    """
    cprint('\n/!\\ Initialization of EpyNN model failed - debug', 'red', attrs=['bold'])

    try:
        # Identify faulty layer
        layer = [layer for layer in model.layers if hasattr(layer, 'check')][0]

        # Update shapes from existing caches
        layer.update_shapes(layer.fc, layer.fs)
        layer.update_shapes(layer.bc, layer.bs)

        # Report debug information for faulty layer
        cprint('%s layer: ' % layer.name, 'red', attrs=['bold'])

        cprint('Known dimensions', 'white', attrs=['bold'])
        print(', '.join([k + ': ' + str(v) for k, v in layer.d.items()]))

        cprint('Known forward shapes', 'green', attrs=['bold'])
        print('\n'.join([k + ': ' + str(v) for k, v in layer.fs.items()]))

        cprint('Known backward shape', 'cyan', attrs=['bold'])
        print('\n'.join([k + ': ' + str(v) for k, v in layer.bs.items()]))

    except:
        pass

    # Report traceback of error and exit program
    cprint('System trace', 'red', attrs=['bold'])
    print(trace)

    sys.exit()
