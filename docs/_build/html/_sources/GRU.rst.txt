.. EpyNN documentation master file, created by
   sphinx-quickstart on Tue Jul  6 18:46:11 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::

Gated Recurrent Unit (GRU)
===============================

Layer architecture
------------------------------

.. image::

.. autoclass:: nnlibs.gru.models.GRU
    :show-inheritance:

    .. automethod:: nnlibs.gru.models.GRU.compute_shapes

        .. literalinclude:: ./../nnlibs/gru/parameters.py
            :pyobject: gru_compute_shapes
            :language: python

    .. automethod:: forward

        .. literalinclude:: ./../nnlibs/gru/forward.py
            :pyobject: gru_forward


    .. automethod:: backward

        .. literalinclude:: ./../nnlibs/gru/backward.py
            :pyobject: gru_backward


    .. automethod:: compute_gradients

        .. literalinclude:: ./../nnlibs/gru/parameters.py
            :pyobject: gru_compute_gradients



Live examples
------------------------------
