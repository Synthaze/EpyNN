.. EpyNN documentation master file, created by
   sphinx-quickstart on Tue Jul  6 18:46:11 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::

Long Short-Term Memory (LSTM)
===============================

Layer architecture
------------------------------

.. image::

.. autoclass:: nnlibs.lstm.models.LSTM
    :show-inheritance:

    .. automethod:: nnlibs.lstm.models.LSTM.compute_shapes

        .. literalinclude:: ./../nnlibs/lstm/parameters.py
            :pyobject: lstm_compute_shapes
            :language: python

    .. automethod:: forward

        .. literalinclude:: ./../nnlibs/lstm/forward.py
            :pyobject: lstm_forward


    .. automethod:: backward

        .. literalinclude:: ./../nnlibs/lstm/backward.py
            :pyobject: lstm_backward


    .. automethod:: compute_gradients

        .. literalinclude:: ./../nnlibs/lstm/parameters.py
            :pyobject: lstm_compute_gradients



Live examples
------------------------------
