# EpyNN/nnlibs/commons/models.py
# Standard library imports
import time

# Related third party imports
import numpy as np

# Local application/library specific imports
import nnlibs.commons.schedule as cs
import nnlibs.commons.maths as cm


class Layer:
    """
    Definition of a Layer meta-type

    Attributes
    ----------
    d : dict
        Dimensions.
    p : dict
        Parameters.
    g : dict
        Gradients.
    fc : dict
        Forward cache.
    bc : dict
        Backward cache.
    fs : dict
        Forward shapes.
    bs : dict
        Backward shapes.
    o : dict
        Other information.
    se_hPars: dict
        Local hyperparameters.
    activation : dict
        Activation functions as key-value pairs for log purpose.
    """

    def __init__(self):

        ## Dimensions
        self.d = {}
        ## Parameters
        self.p = {}
        ## Gradients
        self.g = {}
        ## Forward pass cache
        self.fc = {}
        ## Backward pass cache
        self.bc = {}
        ## Forward pass shapes
        self.fs = {}
        ## Backward pass shapes
        self.bs = {}
        ##
        self.o = {}

        self.se_hPars = None

        self.activation = {}

    def update_shapes(self,mode):

        if mode == 'forward':
            self.fs.update({ k:v.shape for k,v in self.fc.items() })

        elif mode == 'backward':
            self.bs.update({ k:v.shape for k,v in self.bc.items() })


class dataSet:
    """
    Definition of a dataSet object prototype

    Attributes
    ----------
    . : .
        .
    """

    def __init__(self, dataset, name='dummy'):

        # Prepare X data
        x_data = [ x[0] for x in dataset ]
        # Prepare Y data
        y_data = [ x[1] for x in dataset ]

        # Identifier
        self.n = name

        # Set X and Y data for training in corresponding uppercase attributes
        self.X = np.array(x_data)
        self.Y = np.array(y_data)

        # Restore Y data as single digit labels
        self.y = np.argmax(self.Y,axis=1)

        # Labels balance
        self.b = {label:np.count_nonzero(self.y == label) for label in self.y}

        # Set numerical id for each sample
        self.id = np.array([ i for i in range(len(dataset)) ])

        # Number of samples
        self.s = str(len(self.id))

        ## Predicted data
        # Output of forward propagation
        self.A = None
        # Predicted labels
        self.P = None
