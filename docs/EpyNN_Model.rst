.. EpyNN documentation master file, created by
   sphinx-quickstart on Tue Jul  6 18:46:11 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::



Neural Network (Model)
===============================

Neural Network
------------------------------

EpyNN model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: nnlibs.meta.models.EpyNN

    .. automethod:: __init__

    .. automethod:: forward

        .. literalinclude:: ./../nnlibs/meta/forward.py
            :pyobject: model_forward

    .. automethod:: backward

        .. literalinclude:: ./../nnlibs/meta/backward.py
            :pyobject: model_backward

    .. automethod:: train

        .. literalinclude:: ./../nnlibs/meta/train.py
            :pyobject: model_training

    .. automethod:: initialize

    .. automethod:: compute_metrics

    .. automethod:: evaluate

    .. automethod:: logs

    .. automethod:: plot

    .. automethod:: predict


Model settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: nnlibs.settings.config

.. literalinclude:: ../nnlibs/settings.py
    :language: python
    :start-after: GENERAL CONFIGURATION SETTINGS
    :lines: 1-30


.. autoclass:: nnlibs.settings.hPars


.. literalinclude:: ../nnlibs/settings.py
    :language: python
    :start-after: HYPERPARAMETERS SETTINGS
    :lines: 1-22
