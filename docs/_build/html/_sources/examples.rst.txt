.. EpyNN documentation master file, created by
   sphinx-quickstart on Tue Jul  6 18:46:11 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::

Live examples
==================

Live examples in ``EpyNN/nnlive`` are directories each containing the required codes to run dummy and real world examples.

Examples are intended to introduce different input data types, processing schemes and appropriate architecture layers to build Neural Networks with EpyNN.

In the philosophy or EpyNN, example directories are specific templates with respect to the nature of input data. Thus, users can adapt relevant directories for their own situation, with minimal code modification.

Dummy
-------------

Dummy examples are intended to help in understanding processing schemes for input data of diverse natures.

During development or for educational purpose, these examples can also be used for debugging because dummy data are generated from models known *a priori*.

Boleans
~~~~~~~~~~~~~

``EpyNN/nnlive/dummy_bolean``

*Data structure, shape, and preparation*

.. include:: ./../nnlive/dummy_bolean/sets_prepare.py
   :literal:

*Architecture layers, training and prediction*

.. include:: ./../nnlive/dummy_bolean/train.py
   :literal:

Numerical
~~~~~~~~~~~~~

``EpyNN/nnlive/to_be_done``

*Data structure, shape, and preparation*

*Architecture layers, training and prediction*

Strings
~~~~~~~~~~~~~

``EpyNN/nnlive/dummy_strings``

*Data structure, shape, and preparation*

.. include:: ./../nnlive/dummy_strings/sets_prepare.py
   :literal:

*Architecture layers, training and prediction*

.. include:: ./../nnlive/dummy_strings/train.py
   :literal:

Time series
~~~~~~~~~~~~~

``EpyNN/nnlive/dummy_time``

*Data structure, shape, and preparation*

.. include:: ./../nnlive/dummy_time/sets_prepare.py
   :literal:

*Architecture layers, training and prediction*

.. include:: ./../nnlive/dummy_time/train.py
   :literal:

Images
~~~~~~~~~~~~~

``EpyNN/nnlive/dummy_images``

*Data structure, shape, and preparation*

*Architecture layers, training and prediction*

Real
-----------------

Disease
~~~~~~~~~~~~~

``EpyNN/nnlive/to_be_done``

*Data structure, shape, and preparation*

*Architecture layers, training and prediction*

Numerical?
~~~~~~~~~~~~~

``EpyNN/nnlive/to_be_done``

*Data structure, shape, and preparation*

*Architecture layers, training and prediction*

PTMs
~~~~~~~~~~~~~

``EpyNN/nnlive/ptm_prediction``

*Data structure, shape, and preparation*

*Architecture layers, training and prediction*

Music
~~~~~~~~~~~~~

``EpyNN/nnlive/music_author``

*Data structure, shape, and preparation*

*Architecture layers, training and prediction*

MNIST
~~~~~~~~~~~~~

``EpyNN/nnlive/mnist_database``

*Data structure, shape, and preparation*

*Architecture layers, training and prediction*
