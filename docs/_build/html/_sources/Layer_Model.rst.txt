.. EpyNN documentation master file, created by
   sphinx-quickstart on Tue Jul  6 18:46:11 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::

Architecture Layers (Model)
===============================

Layers
------------------------------

Base Layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: nnlibs.commons.models.Layer

    .. automethod:: __init__

    .. automethod:: update_shapes

Layer Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: nnlibs.template.models.Template
    :show-inheritance:

    .. automethod:: __init__

    .. automethod:: compute_shapes

    .. automethod:: initialize_parameters

    .. automethod:: forward

    .. automethod:: backward

    .. automethod:: compute_gradients

    .. automethod:: update_parameters


Layer settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data
------------------------------

Dataset Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: nnlibs.commons.models.dataSet
    :members:

    .. automethod:: __init__

Data embedding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: nnlibs.settings.dataset


.. literalinclude:: ../nnlibs/settings.py
    :language: python
    :start-after: DATASET SETTINGS
    :lines: 1-22
