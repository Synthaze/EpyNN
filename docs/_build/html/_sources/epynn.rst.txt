.. EpyNN documentation master file, created by
   sphinx-quickstart on Tue Jul  6 18:46:11 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::

EpyNN Model
===============================

Network model
------------------------------

.. autoclass:: nnlibs.meta.models.EpyNN
    :members:

    .. automethod:: __init__

    .. automethod:: forward

    .. automethod:: backward

    .. automethod:: initialize

    .. automethod:: train

    .. automethod:: compute_metrics

    .. automethod:: evaluate

    .. automethod:: logs

    .. automethod:: plot

    .. automethod:: predict


Layer model
------------------------------

Base layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: nnlibs.commons.models.Layer

    .. automethod:: __init__

    .. automethod:: update_shapes

Template layer
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



Data model
------------------------------


.. autoclass:: nnlibs.commons.models.dataSet
    :members:

    .. automethod:: __init__

Settings
------------------------------
