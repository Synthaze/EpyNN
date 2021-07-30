.. EpyNN documentation master file, created by
   sphinx-quickstart on Tue Jul  6 18:46:11 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::

.. _Workflows:

Workflows
==================

The default template EpyNN directory contains a ``sets_prepare.py`` file to prepare data sets and a ``train.py`` file to import data, to build and train your model and to use it on unlabeled data.

.. _Initialization:

Initialization
----------------

This subsection presents imports and headers typically found in the ``train.py`` template, as well as the ``settings.py`` file containing EpyNN environment variables.

.. _Imports:

Full imports
~~~~~~~~~~~~~~~~

.. literalinclude:: ./../nnlive/template/train.py
    :lines: 1-25

Upon execution of ``nnlibs.initialize import *``, a default ``settings.py`` file is imported in the current directory if not present.

Other imports are described in the code comments.

.. _Headers:

Headers
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Do not truncate NumPy arrays on print() call - Please comment for large arrays
    np.set_printoptions(precision=3,threshold=sys.maxsize)

    # Set NumPy behavior for floating-point errors.
    # https://numpy.org/doc/1.20/reference/generated/numpy.seterr.html
    np.seterr(all='warn')

    # Set global seed for pseudo-random number generators
    cm.global_seed(1)

    # Create default ./models and ./sets directory to store data
    # Will empty these locations if settings.cfg.directory_clear is set to True
    cl.init_dir(se.config)

    # Initialize runData object
    runData = runData(se.config)

    # Initialize hPars object
    hPars = hPars(se.hPars)

.. _Settings:

Settings
~~~~~~~~~~~~~~~~

.. literalinclude:: ./../nnlive/template/settings.py

Datasets
-------------

Preparation
~~~~~~~~~~~~~~~~

Since data sets preparation is the most variable part of the workflow, it is appropriate to use a separate file ``sets_prepare.py`` containing data-specific codes.

.. literalinclude:: ./../nnlive/template/sets_prepare.py

Import
~~~~~~~~~~~~~~~~

Data sets can then be retrieved in ``train.py`` by calling the ``sets_prepare()`` function from the ``sets_prepare.py`` file located in the working directory.

.. code-block:: python

    dsets = sp.sets_prepare(runData)



Model Builder
-----------------

Stack layers
~~~~~~~~~~~~~~~~

Neural Networks can be assembled easily by stacking layer architectures in a ``list`` object.

.. code-block:: python

    # Stack layers appropriately
    layers = [LSTM(100),Flatten(),Dense(16,cm.elu),Dense(2)]


Initialize your network
~~~~~~~~~~~~~~~~~~~~~~~~

Neural Networks can be initialized by providing the meta-model ``EpyNN`` with the list of layer architectures.

.. code-block:: python

    # Set a name for your model (optional)
    name = 'LSTM_Flatten_Dense_Dense'

    # Initialize your Neural Network
    model = EpyNN(name=name,layers=layers,hPars=hPars)


Model training
-----------------

Train
~~~~~~~~~~~~~~~~

.. code-block:: python

    model.train(dsets,hPars,runData)


Plot results
~~~~~~~~~~~~~~~~

.. code-block:: python

    model.plot(hPars,runData)


Use your model
-----------------

Prepare data
~~~~~~~~~~~~~~~~

.. code-block:: python

    None

Predict
~~~~~~~~~~~~~~~~

.. code-block:: python

    None
