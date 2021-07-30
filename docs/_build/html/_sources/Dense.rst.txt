.. EpyNN documentation master file, created by
   sphinx-quickstart on Tue Jul  6 18:46:11 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::

Fully Connected (Dense)
===============================

Layer architecture
------------------------------

.. image::

.. autoclass:: nnlibs.dense.models.Dense
    :show-inheritance:

    .. automethod:: nnlibs.dense.models.Dense.compute_shapes

        .. literalinclude:: ./../nnlibs/dense/parameters.py
            :pyobject: dense_compute_shapes
            :language: python

    .. automethod:: forward

        .. literalinclude:: ./../nnlibs/dense/forward.py
            :pyobject: dense_forward


    .. automethod:: backward

        .. literalinclude:: ./../nnlibs/dense/backward.py
            :pyobject: dense_backward


    .. automethod:: compute_gradients

        .. literalinclude:: ./../nnlibs/dense/parameters.py
            :pyobject: dense_compute_gradients



Live examples
------------------------------
