.. EpyNN documentation master file, created by
   sphinx-quickstart on Tue Jul  6 18:46:11 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::

Flatten
===============================

Layer architecture
------------------------------

.. image::

.. autoclass:: nnlibs.flatten.models.Flatten
    :show-inheritance:

    .. automethod:: nnlibs.flatten.models.Flatten.compute_shapes

        .. literalinclude:: ./../nnlibs/flatten/parameters.py
            :pyobject: flatten_compute_shapes
            :language: python

    .. automethod:: forward

        .. literalinclude:: ./../nnlibs/flatten/forward.py
            :pyobject: flatten_forward


    .. automethod:: backward

        .. literalinclude:: ./../nnlibs/flatten/backward.py
            :pyobject: flatten_backward


    .. automethod:: compute_gradients

        .. literalinclude:: ./../nnlibs/flatten/parameters.py
            :pyobject: flatten_compute_gradients



Live examples
------------------------------
