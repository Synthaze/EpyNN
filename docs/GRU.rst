.. EpyNN documentation master file, created by
   sphinx-quickstart on Tue Jul  6 18:46:11 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::

Gated Recurrent Unit (GRU)
===============================

Layer architecture
------------------------------

.. image:: _static/gru/gru1-01.svg

.. autoclass:: nnlibs.gru.models.gru
    :show-inheritance:

    .. automethod:: nnlibs.gru.models.gru.compute_shapes

        .. literalinclude:: ./../nnlibs/gru/parameters.py
            :pyobject: gru_compute_shapes
            :language: python

    .. automethod:: forward

        .. literalinclude:: ./../nnlibs/gru/forward.py
            :pyobject: gru_forward

        .. image:: _static/gru/gru2-01.svg

        .. math:: X = A \tag{1}

        .. math:: X_s = X[:, s] \tag{2s}

        .. math:: h_s = h_{act}(W_x \cdot X_s + W_h \cdot h_{s-1} + b_h) \tag{3s}

        .. math:: A_s = A_{act}(W \cdot h_s + b) \tag{4s}


    .. automethod:: backward

        .. literalinclude:: ./../nnlibs/gru/backward.py
            :pyobject: gru_backward

        .. image:: _static/gru/gru3-01.svg

        .. math:: dX = dA \tag{1}

        .. math:: dX_s = dX[s] \tag{2s}

        .. math:: dh_s = h_{act}'(h_s) \times (W.T \cdot dX_s + dh_{s+1}) \tag{3s}

        .. math:: dh_{s+1} = W_h.T \cdot dh_s \tag{4s}

        .. math:: dA_{s} = W_x.T \cdot dh_s \tag{5s}

    .. automethod:: compute_gradients

        .. literalinclude:: ./../nnlibs/gru/parameters.py
            :pyobject: gru_compute_gradients


        .. math::
          \begin{align}
            dW_s &= dX_s \cdot h_s  \\
            db_s &= \sum_{j = 1}^n dX_{s_{ij}}  \tag{A}
          \end{align}

        .. math::
          \begin{align}
            dW_{x_s} &= dh_s \cdot X_s  \\
            dW_{h_s} &= dh_s \cdot h_{s-1} \\
            db_{h_s} &= \sum_{j = 1}^n dh_{s_{ij}} \tag{B}
          \end{align}



Live examples
------------------------------

Dummy string data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. toctree::
    :maxdepth: 1

    gru_binary.ipynb
