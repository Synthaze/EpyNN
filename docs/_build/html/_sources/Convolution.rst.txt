.. EpyNN documentation master file, created by
   sphinx-quickstart on Tue Jul  6 18:46:11 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::

Convolutional Neural Network (CNN)
===================================

Convolution layer
-----------------------------------

Layer architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. autoclass:: nnlibs.convolution.models.Convolution
    :show-inheritance:

    .. automethod:: nnlibs.convolution.models.Convolution.compute_shapes

        .. literalinclude:: ./../nnlibs/convolution/parameters.py
            :pyobject: convolution_compute_shapes
            :language: python

    .. automethod:: forward

        .. literalinclude:: ./../nnlibs/convolution/forward.py
            :pyobject: convolution_forward


    .. automethod:: backward

        .. literalinclude:: ./../nnlibs/convolution/backward.py
            :pyobject: convolution_backward


    .. automethod:: compute_gradients

        .. literalinclude:: ./../nnlibs/convolution/parameters.py
            :pyobject: convolution_compute_gradients


Pooling layer
-----------------------------------

Layer architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. autoclass:: nnlibs.pooling.models.Pooling
    :show-inheritance:

    .. automethod:: nnlibs.pooling.models.Convolution.compute_shapes

        .. literalinclude:: ./../nnlibs/pooling/parameters.py
            :pyobject: pooling_compute_shapes
            :language: python

    .. automethod:: forward

        .. literalinclude:: ./../nnlibs/pooling/forward.py
            :pyobject: pooling_forward


    .. automethod:: backward

        .. literalinclude:: ./../nnlibs/pooling/backward.py
            :pyobject: pooling_backward


    .. automethod:: compute_gradients

        .. literalinclude:: ./../nnlibs/pooling/parameters.py
            :pyobject: pooling_compute_gradients


Live examples
------------------------------
