.. EpyNN documentation master file, created by
   sphinx-quickstart on Tue Jul  6 18:46:11 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::



Neural Network (Model)
===============================

EpyNN Class Object
------------------------------


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


Hyperparameters
------------------------------

.. autoclass:: nnlibs.settings.config

.. autoclass:: nnlibs.settings.se_hPars

.. literalinclude:: ../nnlibs/settings.py
    :language: python
    :start-after: HYPERPARAMETERS SETTINGS
