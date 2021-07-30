.. EpyNN documentation master file, created by
   sphinx-quickstart on Tue Jul  6 18:46:11 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::

Data Embedding (Input)
===============================

Layer architecture
------------------------------

.. image::

.. autoclass:: nnlibs.embedding.models.Embedding
    :show-inheritance:

    .. automethod:: nnlibs.embedding.models.Embedding.compute_shapes

        .. literalinclude:: ./../nnlibs/embedding/parameters.py
            :pyobject: embedding_compute_shapes
            :language: python

    .. automethod:: forward

        .. literalinclude:: ./../nnlibs/embedding/forward.py
            :pyobject: embedding_forward


    .. automethod:: backward

        .. literalinclude:: ./../nnlibs/embedding/backward.py
            :pyobject: embedding_backward


    .. automethod:: compute_gradients

        .. literalinclude:: ./../nnlibs/embedding/parameters.py
            :pyobject: embedding_compute_gradients



Live examples
------------------------------
