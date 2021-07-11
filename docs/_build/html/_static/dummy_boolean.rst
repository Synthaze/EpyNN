
Boolean data type is a form of data with only two possible values, namely ``True`` and ``False`` in most programming languages. In Python, these values evaluate to ``1`` and ``0`` behind the scene. Calculations using Boolean data are very quick and performance gain also arise from easier data and output processing compared to other data types.

Examples of real world topics well suited for the Boolean data type may include: molecular interactions, gene regulation, disease prediction and diagnosis, among others. See the :ref:`Disease` example in the next section.

The directory ``EpyNN/nnlive/dummy_boolean`` contains a dummy example of EpyNN workflow for binary classification tasks with Boolean input data.

*Data preparation, structure and shape*

.. raw:: html

   <details>
   <summary style="cursor:pointer">Expand text details</summary>

   <br>

One ``dataset`` is a list of ``N_SAMPLES`` samples.

One ``sample`` is a list with ``features`` and associated ``label``.

One ``features`` object is a list of Boolean data and ``label`` a one-hot encoded label.

In the ``sets_prepare()`` function, a ``for`` loop iterates over ``N_SAMPLES`` to generate ``features`` list of ``N_FEATURES`` random Boolean elements.

If the arbitrary condition ``( features.count(True) > features.count(False) )`` evaluates to ``True``, then ``features`` is associated with a positive label ``p_label`` in one ``sample`` object.

Conversely, if the opposite condition ``( features.count(True) < features.count(False) )`` evaluates to ``True``, then ``features`` is associated with a negative label ``n_label`` in one ``sample`` object.

At the end of each iteration, ``sample`` is appended to the ``dataset`` list object.

When the ``for`` loop completes, the ``dsets`` list object is returned from the ``cio.dataset_to_dsets()`` function which takes ``dataset`` as argument.

The list ``dsets`` contains training, testing and validation sets such as ``[dtrain,dtest,dval]``.

Elements in ``dsets`` are ``nnlibs.commons.models.dataSet`` objects with ``features`` and ``label`` vectors assigned to ``X`` and ``Y`` attributes, respectively.

.. raw:: html

   </details>

   <br>

.. literalinclude:: ./../nnlive/dummy_boolean/sets_prepare.py
   :end-before: # DOCS_END

.. raw:: html

      <details>
      <summary style="cursor:pointer">Data and shapes</summary>

      <br>

.. literalinclude:: ./_static/dummy_boolean_data_shapes.dat


.. raw:: html

      </details>

      <br>

.. note::

    In real world problems, the relationship between ``features`` (X) and ``label`` (Y) is unknown *a priori*. In this dummy example, the relationship is known *a priori* and the Neural Network should converge to approximate the ``model_dummy_boolean()`` function.

    .. code-block:: python

        def model_dummy_boolean(X):

            if X.count(True) > X.count(False):
                Y = [1,0] # Positive

            elif X.count(True) < X.count(False):
                Y = [0,1] # Negative

            return Y

*Architecture layers, training and prediction*

.. raw:: html

    <details>
    <summary style="cursor:pointer">Imports and headers</summary>

    <br>

In the current example script template, the ``# VARIABLE`` comment in the ``IMPORTS`` part relates to imports which depends on input data type and shape.

The ``HEADERS`` part depends on user preferences and loads settings from the ``settings.py`` file in the working directory.

Details can be found the :ref:`Workflows` chapter under the :ref:`Initialization` section. See :ref:`Imports`, :ref:`Headers` and :ref:`Settings` for direct access.


.. literalinclude:: ./../nnlive/dummy_boolean/train.py
    :end-before: # DOCS_HEADERS


.. raw:: html

    </details>

    <br>

Upon execution of the ``IMPORTS`` and ``HEADERS`` parts, data are returned from the ``sp.sets_prepare()`` function in the ``DATASETS`` part.

The ``BUILD MODEL`` part contains examples of layer architectures and stacks relevant to input data type and shape.

Given a number of samples ``N_SAMPLES`` and number of Boolean features ``N_FEATURES`` describing each sample, the features vector of the corresponding set in this example has shape ``(N_SAMPLES, N_FEATURES)``.

Given EpyNN built-in layer architectures, this shape is suited to the ``Dense`` layer, often referred to as *Fully connected layer*.







.. literalinclude:: ./../nnlive/dummy_boolean/train.py
    :start-after: # DOCS_HEADERS
