.. EpyNN documentation master file, created by
   sphinx-quickstart on Tue Jul  6 18:46:11 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::

Introduction
===============================

The aim of this section is to present basics of Neural Networks in a way that will be accessible for most visitors.

Basic Concepts and Limitations
-------------------------------------

What is a Neural Network?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the literature, it is often explained that Artificial Neural Networks (ANN) evolved from the idea of simulating the human brain :cite:p:`zou2008overview`.

This idea may originate from McCulloch & Pitts who published in 1943 a paper entitled **"A logical calculus of the ideas immanent in nervous activity"** :cite:p:`mcculloch1943logical` where they developed the theory according to which *"Because of the “all-or-none” character of nervous activity, neural events and the relations among them can be treated by means of propositional logic."*

In common language, this means that upon integration and weighting of various stimuli by a given biological neuron, this same neuron may spike or may not spike, thus propagating forward the information to the next neuron.

In ANNs, we would translate by saying that a node within the network will integrate inputs, weight them and pass the product through a ``step_function(input * weight)`` which propagates forward ``True`` or ``False`` to the next node.


Why using Neural Networks?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Neural Networks are claimed to be universal function approximators :cite:p:`hornik1989multilayer`.

In the scope of Supervised Machine Learning (SML), it means that ANNs can theoretically determine any function which will approximate the relationship between inputs (X) and output (Y) known *a priori* and describing a given **sample**.

Importantly, ANNs deal with n-dimensional input **features**. It means that one **sample** is described by one or more features to which a **label** is associated.

Included in a training data set, a relatively large number of samples is provided to the Neural Network which will iterate over a user-provided number of **epochs** or iterations.

Provided with a learning rule, which is generally composed of a **learning rate** and a **cost function** to be minimized, ANNs can auto-adjust their **trainable parameters** to find one function which will make the link between features and labels.

The ability of ANNs to detect patterns in sample features that are relevant to associated label makes them widely used for classification task, anomaly detection, time series predictions, deconvolution and among many others.

Daily life examples may include music source separation which is the task of separating a waveform in individual sources corresponding to voices and accompaniments. Such task was historically challenging but enormous progress were made through deep learning approaches :cite:p:`uhlich2017improving`. Recently, a simple python library was published based on ANNs which provides for anyone state-of-the-art results in such task :cite:p:`hennequin2020spleeter`.

.. seealso::
    How to prepare raw ``.wav`` files: `Music to NumPy`_.



Another example may consist in artist-based painting classification. From a set of art-painting (features) and associated author (label), researchers could develop an ANN which was effective and achieved state-of-the-art performance in pairing painting and the corresponding author :cite:p:`hua2020artist`.

.. seealso::
    | How to prepare dummy images: `Integers to Image`_.
    | How to prepare handwritten digits images from the MNIST database: `Prepare MNIST`_.


In the daily life of the *biochemist*, the *structural biologist* or more generally anyone interested in protein or peptide sequences, first applications of Neural Networks date from decades. As early as 1989, Howard L. Holley and Nobel Prize laureate Mark Karplus published a paper entitled **"Protein secondary structure prediction with a neural network"** :cite:p:`holley1989protein`. Since then, the *NMR spectroscopist* may acknowledge **"TALOS+: a hybrid method for predicting protein backbone torsion angles from NMR chemical shifts"** :cite:p:`shen2009talos` which partially relies on Neural Networks to predict angle constrains extensively used in protein structure calculation from NMR data. Finally, the field of protein research is looking forward to recent developments from **"Highly accurate protein structure prediction with AlphaFold"** :cite:p:`jumper2021highly`.

.. seealso::
    | How to prepare string data: `String Encoding`_.
    | How to prepare peptide sequences: `Prepare Peptides`_.



.. _Integers to Image: nnlive/dummy_image/prepare_dataset.ipynb

.. _Prepare MNIST: nnlive/dummy_image/prepare_dataset.ipynb

.. _Music to NumPy: nnlive/author_music/prepare_dataset.ipynb

.. _String encoding: nnlive/dummy_string/prepare_dataset.ipynb

.. _Prepare Peptides: nnlive/ptm_protein/prepare_dataset.ipynb


Which limits for Neural Networks?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Artificial Neural Networks (ANNs) are extremely efficient computational models which can solve problems in any fields.

However, there are classical limitations to such models.

* **Training corpus**

In Supervised Machine Learning (SML) based on ANNs, the principle is to provide a training corpus which should contains a large number of samples being representative of the phenomenon of interest.

* **Overfitting**

Overfitting is an error in regression procedure which happens when the model corresponds to closely to a particular set of data. Said differently, ANNs tend to include outliers in the output model.

.. seealso::
    | Example of overfitting on random data

* **Explainability**

Because ANNs are nested and non-linear structures with thousands - up to billions - parameters, they are difficult to interpret and are often considered as "black boxes" used for performance :cite:p:`samek2019towards`.



Implementation and Vocabulary
-------------------------------------

Artificial Neural Networks (ANNs) can be implemented in Python.

Python is an *object-oriented* programming language. This means that abstract objects - equivalent to object prototypes - can be defined is a custom manner by the programmer.

Such prototypes are defined in ``class``, such as:

.. code-block:: python

    class MyModel:

        def __init__(self):
            self.name = 'MyModel'

        def my_method(self, msg):
            print(msg)


The object related to the class ``MyModel`` can be **instantiated** by calling the corresponding **class constructor** such as:

.. code-block:: python

    my_object = MyModel()

The object ``my_object`` is now an **instance** of the class ``MyModel``.

When the instance ``my_object`` is being created, the ``__init__()`` method is executed. Later on, other methods defined within the class can be executed from the instantiated object, such as:

.. code-block:: python

    my_object.my_method('My Message')

Which will print ``My Message`` on the terminal session.

.. seealso::

    | Interactive examples on `W3 Schools`_.
    | Detailed explanations on `Python Official Documentation`_.

.. _W3 Schools: https://www.w3schools.com/python/python_classes.asp
.. _Python Official Documentation: https://docs.python.org/3/tutorial/classes.html

This scheme is the basis for the design of most AI-related libraries, including EpyNN.


Neural Network Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If willing to define a model object - or object prototype - for a Neural Network, then a minimalist version could be summarized such as:

.. code-block:: python

    class MyNetwork:
        """
        This is a non-functional summary.
        """

        def __init__(self, layers=[]):

            self.layers = layers

        def training(self, X_data, Y_data):
            # Procedure to train the network from labeled data
            pass

        def predict(self, X_data):
            # Procedure to predict label from unlabeled data
            pass

The corresponding object instance can be instantiated such as:

.. code-block:: python

    my_network = MyNetwork(layers=[input_layer, output_layer])

While the network could be trained such as:

.. code-block:: python

    my_network.train(X_train, Y_train)

And further used to predict from unlabeled data, such as:

.. code-block:: python

    my_network.predict(X_test)

In short, the Neural Network model itself implements the training procedure. The actual architecture of the Neural Network is defined within the ``layers`` argument passed when calling the class constructor.

The detailed implementation of the ``EpyNN`` model for such Neural Network object is presented in the :doc:`EpyNN_Model` section.


Layer Architecture Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have seen that a model of Neural Network can be instantiated by calling its class constructor provided with a list of layers.

Layers prototypes can also be defined within classes, with a minimal example such as:

.. code-block:: python

    class MyLayer:
        """
        This is a non-functional summary.
        """

        def __init__(self):

        def forward(self, A):
            # Procedure to propagate the signal forward
            return A    # To next layer

        def backward(self, dA):
            # Procedure to propagate the error backward
            return dA    # To previous layer

        def update_parameters(self, gradients):
            # Procedure to update parameters from gradients
            pass

The detailed implementation of the ``Layer`` parent model in EpyNN is presented in the :doc:`EpyNN_Layer` section along with a example of child ``Template(Layer)`` model.


Data Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have seen how a Neural Network object could be instantiated by calling the corresponding class constructor which takes a list of layers as argument, itself representing the custom architecture designed by the end-user.

It is a good practice to define a data model to which ``X_data`` and ``Y_data`` will be linked, together with dependent outputs generated by the Neural Network during training and prediction.

Such minimal model may look like:

.. code-block:: python

    class MyData:
        """
        This is a non-functional summary.
        """

        def __init__(self, X_data, Y_data=None):

            self.X = X_data
            self.Y = Y_data

In practice, such object may be linked with the input or embedding layer.

The detailed implementation of the ``dataSet`` data model in EpyNN is presented in the :doc:`Data_Model` while the implementation of the ``Embedding`` layer model is presented in the :doc:`Embedding` section.


.. bibliography:: refs.bib
    :style: unsrt
