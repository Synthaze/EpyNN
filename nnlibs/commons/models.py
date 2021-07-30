# EpyNN/nnlibs/commons/models.py
# Standard library imports
import time

# Related third party imports
import numpy as np


class Layer:
    """
    Definition of a layer meta-prototype. All classes which define a layer prototype inherit from this class. Layer prototypes are defined with respect to a specific architecture (Dense, RNN, Convolution...) whereas the layer meta-prototype is a parent class defining instance attributes common to every layer object.
    """

    def __init__(self):
        """Initialize instance variable attributes.

        Attributes
        ----------
        d : dict
            Instance attribute for layer **dimensions**. It contains **scalar quantities** such as the number of nodes, hidden units, filters or samples with respect to a given layer prototype and instantiated object.
        fs : dict
            Instance attribute for **forward shapes**. It contains **tuples of integers** defining multi-dimensional shapes related to forward propagation. This includes shapes for parameters (weight, bias) and for layer input, output, as well as processing intermediates.
        p : dict
            Instance attribute for **parameters**. It contains independant :class:`numpy.ndarray` associated with **weight** and **bias** with respect to a given layer architecture. Shapes for parameters initialization are retrieved from the *fs* attribute described above.
        fc : dict
            Instance attribute for **forward cache**. It contains independant :class:`numpy.ndarray` associated with **forward propagation processing** which includes input, output and intermediates.
        bs : dict
            Instance attribute for **backward shapes**. It contains **tuples of integers** defining multi-dimensional shapes related to backward propagation. This includes shapes for gradients (weight, bias) and for layer input, output, as well as processing intermediates.
        g : dict
            Instance attribute for **gradients**. It contains independant :class:`numpy.ndarray` associated with **weight and bias gradients** with respect to parameters from the *p* attribute described above. Shapes for gradients are identical to the corresponding parameters.
        bc : dict
            Instance attribute for **backward cache**. It contains independant :class:`numpy.ndarray` associated with **backward propagation processing** which includes input, output and intermediates.
        o : dict
            Instance attribute for other information. Rarely used. It may contain scalar quantities that do not fit within the above-described dictionary attributes.
        activation : dict
            Activation functions for layer. This conveniency attribute stores key:value pairs for activation gates and name of the corresponding function.
        se_hPars: NoneType
            Hyperparameters for layer. It may contain scalar quantities and string defining layer-specific hyperparameters. Hyperparameters are otherwise defined globally upon instanciation of the :class:`nnlibs.meta.models.EpyNN` meta-model.
        """
        self.d = {}
        self.fs = {}
        self.p = {}
        self.fc = {}
        self.bs = {}
        self.g = {}
        self.bc = {}
        self.o = {}

        self.activation = {}

        self.se_hPars = None

        return None

    def update_shapes(self, cache, shapes):
        """Update shapes from cache.

        :param cache: Forward of backward cache as documented above.
        :type cache: dict

        :param shapes: Corresponding forward or backward shapes as documented above.
        :type shapes: dict
        """
        shapes.update({k:v.shape for k,v in cache.items()})

        return None


class dataSet:
    """
    Definition of a dataSet object prototype
    """

    def __init__(self, dataset, label=True, name='dummy'):
        """.

        :param dataset: .
        :type dataset: list

        :param label: .
        :type label: bool

        :param name: .
        :type name: string

        :ivar n:
        :vartype n:

        :ivar X:
        :vartype X:

        :ivar Y:
        :vartype Y:

        :ivar y:
        :vartype y:

        :ivar b:
        :vartype b:

        :ivar id:
        :vartype id:

        :ivar s:
        :vartype s:

        :ivar A:
        :vartype A:

        :ivar P:
        :vartype P:
        """
        x_data = [ x[0] for x in dataset ]
        y_data = [ x[1] for x in dataset ]

        self.n = name

        self.X = np.array(x_data)
        self.Y = np.array(y_data)

        if label:
            self.y = np.argmax(self.Y,axis=1)

            # Labels balance
            self.b = {label:np.count_nonzero(self.y == label) for label in self.y}

        # Set numerical id for each sample
        self.id = np.array([ i for i in range(len(dataset)) ])

        # Number of samples
        self.s = str(len(self.id))

        self.A = None
        self.P = None

        return None
