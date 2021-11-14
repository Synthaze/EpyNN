# EpyNN/epynn/commons/models.py
# Related third party imports
import numpy as np
 

class Layer:
    """
    Definition of a parent **base layer** prototype. Any given **layer** prototype inherits from this class and is defined with respect to a specific architecture (Dense, RNN, Convolution...). The **parent** base layer defines instance attributes common to any **child** layer prototype.
    """

    def __init__(self, se_hPars=None):
        """Initialize instance variable attributes.

        :ivar d: Layer **dimensions** containing scalar quantities such as the number of nodes, hidden units, filters or samples.
        :vartype d: dict[str, int]

        :ivar fs: Layer **forward shapes** for parameters, input, output and processing intermediates.
        :vartype fs: dict[str, tuple[int]]

        :ivar p: Layer weight and bias **parameters**. These are the trainable parameters.
        :vartype p: dict[str, :class:`numpy.ndarray`]

        :ivar fc: Layer **forward cache** related for input, output and processing intermediates.
        :vartype fc: dict[str, :class:`numpy.ndarray`]

        :ivar bs: Layer **backward shapes** for gradients, input, output and processing intermediates.
        :vartype bs: dict[str, tuple[int]]

        :ivar g: Layer **gradients** used to update the trainable parameters.
        :vartype g: dict[str, :class:`numpy.ndarray`]

        :ivar bc: Layer **backward cache** for input, output and processing intermediates.
        :vartype bc: dict[str, :class:`numpy.ndarray`]

        :ivar o: Other scalar quantities that do not fit within the above-described attributes (rarely used).
        :vartype o: dict[str, int]

        :ivar activation: Conveniency attribute containing names of activation gates and corresponding activation functions.
        :vartype activation: dict[str, str]

        :ivar se_hPars: Layer **hyper-parameters**.
        :vartype se_hPars: dict[str, str or int]
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

        :param cache: Cache from forward or backward propagation.
        :type cache: dict[str, :class:`numpy.ndarray`]

        :param shapes: Corresponding shapes.
        :type shapes: dict[str, tuple[int]]
        """
        shapes.update({k:v.shape for k,v in cache.items()})

        return None


class dataSet:
    """
    Definition of a dataSet object prototype.

    :param X_data: Set of sample features.
    :type X_data: list[list[int or float]]

    :param Y_data: Set of sample label, defaults to None.
    :type Y_data: list[list[int] or int] or NoneType, optional

    :param name: Name of set, defaults to 'dummy'.
    :type name: str, optional
    """

    def __init__(self,
                 X_data,
                 Y_data=None,
                 name='dummy'):
        """Initialize dataSet object.

        :ivar active: `True` if X_data is not empty.
        :vartype active: bool

        :ivar X: Set of sample features.
        :vartype X: :class:`numpy.ndarray`

        :ivar Y: Set of sample label.
        :vartype Y: :class:`numpy.ndarray`

        :ivar y: Set of single-digit sample label.
        :vartype y: :class:`numpy.ndarray`

        :ivar b: Balance of labels in set.
        :vartype b: dict[int: int]

        :ivar ids: Sample identifiers.
        :vartype ids: list[int]

        :ivar A: Output of forward propagation.
        :vartype A: :class:`numpy.ndarray`

        :ivar P: Label predictions.
        :vartype P: :class:`numpy.ndarray`

        :ivar name: Name of dataset.
        :vartype name: str
        """
        self.name = name

        self.active = True if len(X_data) > 0 else False

        # Set of sample features
        if self.active:

            # Vectorize X_data in NumPy array
            self.X = np.array(X_data)

        # Set of sample label
        if (hasattr(Y_data, 'shape') or Y_data) and self.active:

            # Vectorize Y_data in NumPy array
            Y_data = np.array(Y_data)

            # Check if labels are one-hot encoded or not
            is_encoded = True if Y_data.ndim == 2 else False

            # If not encoded, reshape from (N_SAMPLES,) to (N_SAMPLES, 1)
            self.Y = Y_data if is_encoded else np.expand_dims(Y_data, 1)

            # Retrieve single-digit label w. r. t. one-hot encoding
            self.y = np.argmax(Y_data, axis=1) if is_encoded else Y_data

            # Map single-digit label with representation in dataset
            self.b = {label:np.count_nonzero(self.y == label) for label in self.y}

        # Set sample identifiers
        self.ids = np.array([i for i in range(len(X_data))])

        # Initialize empty arrays
        self.A = np.array([])
        self.P = np.array([])

        return None
