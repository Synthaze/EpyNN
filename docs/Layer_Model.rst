.. EpyNN documentation master file, created by
   sphinx-quickstart on Tue Jul  6 18:46:11 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::

Architecture Layers (Model)
===============================

Base Layer
------------------------------


.. autoclass:: nnlibs.commons.models.Layer

    .. automethod:: __init__

    .. automethod:: update_shapes

Template Layer
------------------------------

.. autoclass:: nnlibs.template.models.Template
    :show-inheritance:

    .. automethod:: __init__

    .. automethod:: compute_shapes

        .. literalinclude:: ./../nnlibs/template/parameters.py
            :pyobject: template_compute_shapes

    .. automethod:: initialize_parameters

        .. literalinclude:: ./../nnlibs/template/parameters.py
            :pyobject: initialize_parameters

    .. automethod:: forward

        .. literalinclude:: ./../nnlibs/template/forward.py
            :pyobject: template_forward

    .. automethod:: backward

        .. literalinclude:: ./../nnlibs/template/backward.py
            :pyobject: template_backward

    .. automethod:: compute_gradients

        .. literalinclude:: ./../nnlibs/template/parameters.py
            :pyobject: template_compute_gradients

    .. automethod:: update_parameters

        .. literalinclude:: ./../nnlibs/template/parameters.py
            :pyobject: template_update_parameters
            

Layer Settings
------------------------------
