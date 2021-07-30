.. EpyNN documentation master file, created by
   sphinx-quickstart on Tue Jul  6 18:46:11 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::

Dropout
===============================

Layer architecture
------------------------------

.. image::

.. autoclass:: nnlibs.dropout.models.Dropout
    :show-inheritance:

    .. automethod:: nnlibs.dropout.models.Dropout.compute_shapes

        .. literalinclude:: ./../nnlibs/dropout/parameters.py
            :pyobject: dropout_compute_shapes
            :language: python

    .. automethod:: forward

        .. literalinclude:: ./../nnlibs/dropout/forward.py
            :pyobject: dropout_forward


    .. automethod:: backward

        .. literalinclude:: ./../nnlibs/dropout/backward.py
            :pyobject: dropout_backward


    .. automethod:: compute_gradients

        .. literalinclude:: ./../nnlibs/dropout/parameters.py
            :pyobject: dropout_compute_gradients



Live examples
------------------------------
