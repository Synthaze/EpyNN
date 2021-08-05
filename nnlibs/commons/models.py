# EpyNN/nnlibs/commons/models.py
# Standard library imports
import time

# Related third party imports
import numpy as np


class Layer:
    """
    Definition of a layer meta-prototype. All classes which define a layer prototype inherit from this class. Layer prototypes are defined with respect to a specific architecture (Dense, RNN, Convolution...) whereas the layer meta-prototype is a parent class defining instance attributes common to every layer object.
    """

    def __init__(self, se_hPars=None):
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

        self.se_hPars = se_hPars

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
    Definition of a dataSet object prototype.

    :param dataset: Contains samples
    :type dataset: list[list[list,list[int]]]

    :param label: Set to False if dataset contains unlabeled samples
    :type label: bool

    :param name: For model seeding
    :type name: str
    """

    def __init__(self,
                dataset,
                label=True,
                name='dummy'):
        """Initialize dataSet object.

        :ivar n: Name of dataset
        :vartype n: str

        :ivar X: Features for samples
        :vartype X: :class:`numpy.ndarray`

        :ivar Y: One-hot encoded labels for samples
        :vartype Y: :class:`numpy.ndarray`

        :ivar y: Single-digit labels for samples
        :vartype y: :class:`numpy.ndarray`

        :ivar b: Labels balance in dataset
        :vartype b: dict

        :ivar ids: Numerical identifiers for samples
        :vartype ids: list

        :ivar s: Total number of samples
        :vartype s: int

        :ivar A: Output of forward propagation for dataset
        :vartype A: :class:`numpy.ndarray`

        :ivar P: Predictions for dataset
        :vartype P: :class:`numpy.ndarray`
        """
        if len(dataset) == 0:
            self.active = False

            return None

        self.active = True

        self.name = name

        if label:
            X_dataset = [x[0] for x in dataset]
        else:
            X_dataset = dataset

        self.X = np.array(X_dataset)

        # Set numerical id for each sample
        self.ids = np.array([i for i in range(self.X.shape[0])])
        # Number of samples
        self.s = str(len(self.ids))

        self.A = np.array([])
        self.P = np.array([])

        if not label:

            return None

        Y_dataset = [x[1] for x in dataset]

        self.Y = np.array(Y_dataset)

        if self.Y.ndim == 1:
            self.Y = np.expand_dims(self.Y, 1)

        # Single-digit labels
        self.y = np.argmax(self.Y, axis=1)
        # Labels balance
        self.b = {label:np.count_nonzero(self.y == label) for label in self.y}

        return None
