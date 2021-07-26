# EpyNN/nnlibs/meta/initialize.py
# Standard library imports
import json
import sys

# Related third party imports
from termcolor import cprint

# Local application/library specific imports
from nnlibs.commons.logs import pretty_json


def initialize_model_layers(model):

        embedding = model.layers[0]

        sample = embedding.batch_dtrain[0]

        A = X = sample.X
        Y = sample.Y

        for layer in model.layers:

            model.network[id(layer)]['Layer'] = layer.__class__.__name__
            model.network[id(layer)]['Activation'] = layer.activation
            model.network[id(layer)]['Dimensions'] = layer.d

            layer.compute_shapes(A)

            model.network[id(layer)]['FW_Shapes'] = layer.fs

            layer.initialize_parameters()

            A = layer.forward(A)

            model.network[id(layer)]['FW_Shapes'] = layer.fs

        dA = A - sample.Y.T

        for layer in reversed(model.layers):

            dA = layer.backward(dA)

            model.network[id(layer)]['BW_Shapes'] = layer.bs

            layer.update_gradients()

        return None


def initialize_exceptions(model,trace):

    for layer in model.network.keys():
        pretty_json(model.network[layer])

    cprint('/!\\ Initialization of EpyNN model failed','red',attrs=['bold'])

    print (trace)

    sys.exit()
