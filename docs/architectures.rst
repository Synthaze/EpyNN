.. EpyNN documentation master file, created by
   sphinx-quickstart on Tue Jul  6 18:46:11 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::

Architectures
==================

Layer
~~~~~~~~~~~~~~

.. autoclass:: nnlibs.commons.models.Layer

    .. automethod:: __init__

    .. seealso::

        :py:mod:``
           Documentation of ...

        :py:mod:``
           Documentation of the :class:`nnlibs.meta.models.EpyNN` meta-model.


    .. automethod:: update_shapes



Template
~~~~~~~~~~~~~~

.. autoclass:: nnlibs.template.models.Template
    :show-inheritance:

    .. automethod:: __init__

    .. automethod:: compute_shapes

    .. automethod:: initialize_parameters

    .. automethod:: forward

    .. automethod:: backward

    .. automethod:: update_gradients

    .. automethod:: update_parameters



Feed-forward
--------------

.. _Dense:

Dense
~~~~~~~~~~~~~~

.. autoclass:: nnlibs.dense.models.Dense
    :show-inheritance:



Recurrent
--------------

RNN
~~~~~~~~~~~~~~

.. autoclass:: nnlibs.rnn.models.RNN
    :show-inheritance:

GRU
~~~~~~~~~~~~~~

.. autoclass:: nnlibs.gru.models.GRU
    :show-inheritance:

LSTM
~~~~~~~~~~~~~~

.. autoclass:: nnlibs.lstm.models.LSTM
    :show-inheritance:



Convolutional
--------------

Convolution
~~~~~~~~~~~~~~

.. autoclass:: nnlibs.convolution.models.Convolution
    :show-inheritance:

Pooling
~~~~~~~~~~~~~~

.. autoclass:: nnlibs.pooling.models.Pooling
    :show-inheritance:


Adapter
--------------

Flatten
~~~~~~~~~~~~~~

.. autoclass:: nnlibs.flatten.models.Flatten
    :show-inheritance:

Embedding
~~~~~~~~~~~~~~

.. autoclass:: nnlibs.embedding.models.Embedding
    :show-inheritance:


Regularization
----------------

Dropout
~~~~~~~~~~~~~~

.. autoclass:: nnlibs.dropout.models.Dropout
    :show-inheritance:



Alt
----------------

Dataset
~~~~~~~~~~~~~~

.. autoclass:: nnlibs.commons.models.dataSet


EpyNN
~~~~~~~~~~~~~~


.. autoclass:: nnlibs.meta.models.EpyNN
    :show-inheritance:

    .. automethod:: __init__

    .. automethod:: forward

    .. automethod:: backward

    .. automethod:: initialize

    .. automethod:: train

    .. automethod:: compute_metrics

    .. automethod:: evaluate

    .. automethod:: logs

    .. automethod:: plot
