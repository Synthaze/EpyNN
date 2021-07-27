.. EpyNN documentation master file, created by
   sphinx-quickstart on Tue Jul  6 18:46:11 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::

RNN
==================


.. image:: _static/RNN/rnn-01.svg

.. autoclass:: nnlibs.rnn.models.RNN
    :show-inheritance:

    .. automethod:: compute_shapes

        .. literalinclude:: ./../nnlibs/rnn/parameters.py
            :pyobject: rnn_compute_shapes
            :language: python

    .. automethod:: forward

        .. literalinclude:: ./../nnlibs/rnn/forward.py
            :pyobject: rnn_forward
            :emphasize-lines: 16,17

        .. math:: X = A \tag{1}

        .. math:: X_s = X[:, s] \tag{2s}

        .. math:: h_s = h_{act}(U \cdot X_s + V \cdot h_{s-1} + b_h) \tag{3s}

        .. math:: A_s = A_{act}(W \cdot h_s + b) \tag{4s}


    .. automethod:: backward

        .. literalinclude:: ./../nnlibs/rnn/backward.py
            :pyobject: rnn_backward

        .. math:: dX = dA \tag{6'}

        .. math:: dX_s = dX[s] \tag{5'}

        .. math:: dhx_s = W.T \cdot dX_s + dh_{s+1} \tag{4a'}

        .. math:: dh_s = dhx_s \times h_{act}'(h_s)  \tag{4b'}

        .. math:: dh_{s+1} = Wh.T \cdot dh \tag{3a'}


    .. automethod:: compute_gradients

        .. literalinclude:: ./../nnlibs/rnn/parameters.py
            :pyobject: rnn_compute_gradients


        .. math::
            dW_s &= (m^{-1}) \times (dX_s \cdot h_s)    \\
            db_s &= (m^{-1}) \times\sum_{j = 1}^n dX_{s_{ij}}

        .. math::
            dWx_s &= (m^{-1}) \times (df_s \cdot X_s)    \\
            dWh_s &= (m^{-1}) \times (df_s \cdot h_{s-1})    \\
            dbh_s &= (m^{-1}) \times\sum_{j = 1}^n df_{s_{ij}}
