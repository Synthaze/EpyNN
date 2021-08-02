# EpyNN/nnlibs/meta/initialize.py
# Standard library imports
import json
import sys

# Related third party imports
from termcolor import cprint

# Local application/library specific imports
from nnlibs.commons.logs import pretty_json


def model_initialize(model, init_params=True):
    """Initialize Neural Network.

    :param init_params:
    :type init_params: bool

    :param model: An instance of EpyNN network.
    :type model: :class:`nnlibs.meta.models.EpyNN`
    """
    batch_dtrain = model.embedding.batch_dtrain

    sample = batch_dtrain[0]

    A = X = sample.X
    Y = sample.Y

    for layer in model.layers:

        layer.name = layer.__class__.__name__
        model.network[id(layer)]['Layer'] = layer.name
        model.network[id(layer)]['Activation'] = layer.activation
        model.network[id(layer)]['Dimensions'] = layer.d

        cprint('Layer: ' + layer.name, attrs=['bold'])
        cprint('compute_shapes: ' + layer.name, 'green', attrs=['bold'])
        layer.compute_shapes(A)

        model.network[id(layer)]['FW_Shapes'] = layer.fs

        if init_params:
            cprint('initialize_parameters: ' + layer.name, 'green', attrs=['bold'])
            layer.initialize_parameters()

        cprint('forward: ' + layer.name, 'green', attrs=['bold'])
        A = layer.forward(A)

        model.network[id(layer)]['FW_Shapes'] = layer.fs

    dA = model.training_loss(Y, A, deriv=True) / A.shape[1]

    for layer in reversed(model.layers):

        cprint('backward: ' + layer.name, 'cyan', attrs=['bold'])
        dA = layer.backward(dA)

        model.network[id(layer)]['BW_Shapes'] = layer.bs

        cprint('compute_gradients: ' + layer.name, 'cyan', attrs=['bold'])
        layer.compute_gradients()

    return None


def model_initialize_exceptions(model,trace):
    """Handle error in model initialization and show logs.
    """
    for layer in model.network.keys():
        pretty_json(model.network[layer])

    cprint('/!\\ Initialization of EpyNN model failed','red',attrs=['bold'])

    print (trace)

    sys.exit()
