.. EpyNN documentation master file, created by
   sphinx-quickstart on Tue Jul  6 18:46:11 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::

Architectures
==================

Feed-forward
--------------

Dense
~~~~~~~~~~~~~~

.. automodule:: nnlibs.dense.models
   :members:

.. autofunction:: nnlibs.dense.forward.dense_forward


.. autofunction:: nnlibs.dense.backward.dense_backward

.. literalinclude:: ./../nnlibs/dense/parameters.py
    :pyobject: init_params

Recurrent
--------------

RNN
~~~~~~~~~~~~~~

.. automodule:: nnlibs.rnn.models
   :members:

.. autofunction:: nnlibs.rnn.forward.rnn_forward


.. autofunction:: nnlibs.rnn.backward.rnn_backward

.. literalinclude:: ./../nnlibs/rnn/parameters.py
    :pyobject: init_params

GRU
~~~~~~~~~~~~~~

.. automodule:: nnlibs.gru.models
   :members:

.. autofunction:: nnlibs.gru.forward.gru_forward


.. autofunction:: nnlibs.gru.backward.gru_backward

.. literalinclude:: ./../nnlibs/gru/parameters.py
    :pyobject: init_params

LSTM
~~~~~~~~~~~~~~

.. automodule:: nnlibs.lstm.models
   :members:

.. autofunction:: nnlibs.lstm.forward.lstm_forward


.. autofunction:: nnlibs.lstm.backward.lstm_backward

.. literalinclude:: ./../nnlibs/lstm/parameters.py
    :pyobject: init_params


Convolutional
--------------
Conv
~~~~~~~~~~~~~~

.. automodule:: nnlibs.conv.models
   :members:

.. autofunction:: nnlibs.conv.forward.convolution_forward


.. autofunction:: nnlibs.conv.backward.convolution_backward

.. literalinclude:: ./../nnlibs/conv/parameters.py
    :pyobject: init_params

Pool
~~~~~~~~~~~~~~

.. automodule:: nnlibs.pool.models
   :members:

.. autofunction:: nnlibs.pool.forward.pooling_forward


.. autofunction:: nnlibs.pool.backward.pooling_backward


Adapter
--------------

Flatten
~~~~~~~~~~~~~~

.. automodule:: nnlibs.flatten.models
   :members:

.. autofunction:: nnlibs.flatten.forward.flatten_forward


.. autofunction:: nnlibs.flatten.backward.flatten_backward



Regularization
----------------

Dropout
~~~~~~~~~~~~~~

.. automodule:: nnlibs.dropout.models
   :members:

.. autofunction:: nnlibs.dropout.forward.dropout_forward


.. autofunction:: nnlibs.dropout.backward.dropout_backward
