.. EpyNN documentation master file, created by
   sphinx-quickstart on Tue Jul  6 18:46:11 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::

Organization
==================

Guidelines
---------------
Although EpyNN can be used for production, it is meant to be a library of homogeneous architecture templates and practical examples which is expected to save an important amount of time for people who wish to learn, teach or develop from scratch, by themselves.

Pure Python/NumPy
~~~~~~~~~~~~~~~~~~
EpyNN in written in pure Python/NumPy without a single import from AI librairies.

By focusing on readability, the source code of EpyNN thus directly reflects the mathematics underlying the diverse architectures provided with EpyNN.

Functions over Class Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Although Python is well suited - if not designed - for extensive use of Class and Method attributes, we choose to favor Functions as much as possible, consistently with directory structures.

Less intricate code
~~~~~~~~~~~~~~~~~~~~
The root ``EpyNN/`` directory contains the ``nnlibs`` sub-directory which includes commons, the EpyNN meta-model, and each layer architectures used to build Neural Networks.

Consequently, a single level of recursion from ``EpyNN/nnlibs`` is required to explore all content and associated python sources.



Layers
---------------
Neural Networks are made of architecture layers, implemented in EpyNN as **models**.

A single layer contains nodes - often refereed to as neurons - and architecture-specific **forward** and **backward** schemes for *signal* and *error* propagation, respectively.

Whereas a forward-propagated signal yields a result, a backward-propagated error enables the Neural Network to learn by computing *gradients* used to alter architecture-specific **parameters**.

models
~~~~~~~~~~~~~~~
Each layer directory contains a ``models.py`` file.

.. include:: ./../nnlibs/template/models.py
   :literal:

forward
~~~~~~~~~~~~~~~
Each layer directory contains a ``forward.py`` file.

.. include:: ./../nnlibs/template/forward.py
   :literal:

backward
~~~~~~~~~~~~~~~
Each layer directory contains a ``backward.py`` file.

.. include:: ./../nnlibs/template/backward.py
   :literal:

parameters
~~~~~~~~~~~~~~~
Each layer directory contains a ``parameters.py`` file.

.. include:: ./../nnlibs/template/parameters.py
   :literal:


Meta
---------------
models
~~~~~~~~~~~~~~~

.. include:: ./../nnlibs/meta/models.py
   :literal:

forward
~~~~~~~~~~~~~~~

.. include:: ./../nnlibs/meta/forward.py
   :literal:

backward
~~~~~~~~~~~~~~~

.. include:: ./../nnlibs/meta/backward.py
   :literal:

parameters
~~~~~~~~~~~~~~~

.. include:: ./../nnlibs/meta/parameters.py
   :literal:

train
~~~~~~~~~~~~~~~

.. include:: ./../nnlibs/meta/train.py
   :literal:



Commons
---------------
decorators
~~~~~~~~~~~~~~~
io
~~~~~~~~~~~~~~~

.. include:: ./../nnlibs/commons/io.py
   :literal:

library
~~~~~~~~~~~~~~~

logs
~~~~~~~~~~~~~~~

maths
~~~~~~~~~~~~~~~

.. literalinclude:: ./../nnlibs/commons/maths.py
   :lines: 1-6

**Weights initialization**

* Xavier

.. literalinclude:: ./../nnlibs/commons/maths.py
    :pyobject: xavier

* Orthogonal

.. literalinclude:: ./../nnlibs/commons/maths.py
    :pyobject: orthogonal

**Activation functions and derivatives**

*Rectifier Linear Unit (ReLU)*

* Primary

.. literalinclude:: ./../nnlibs/commons/maths.py
    :pyobject: relu

* Derivative

.. literalinclude:: ./../nnlibs/commons/maths.py
    :pyobject: drelu

*Leaky ReLu (LReLU)*

* Primary

.. literalinclude:: ./../nnlibs/commons/maths.py
    :pyobject: lrelu

* Derivative

.. literalinclude:: ./../nnlibs/commons/maths.py
    :pyobject: dlrelu

*Exponential Linear Unit (ELU)*

* Primary

.. literalinclude:: ./../nnlibs/commons/maths.py
    :pyobject: elu

* Derivative

.. literalinclude:: ./../nnlibs/commons/maths.py
    :pyobject: delu

*Sigmoid*

* Primary

.. literalinclude:: ./../nnlibs/commons/maths.py
    :pyobject: sigmoid

* Derivative

.. literalinclude:: ./../nnlibs/commons/maths.py
    :pyobject: dsigmoid

*Hyperbolic tangent (tanh)*

* Primary

.. literalinclude:: ./../nnlibs/commons/maths.py
    :pyobject: tanh

* Derivative

.. literalinclude:: ./../nnlibs/commons/maths.py
    :pyobject: dtanh

*Swish*

* Primary

.. literalinclude:: ./../nnlibs/commons/maths.py
    :pyobject: swish

* Derivative

.. literalinclude:: ./../nnlibs/commons/maths.py
    :pyobject: dswish

*Softmax*

* Primary

.. literalinclude:: ./../nnlibs/commons/maths.py
    :pyobject: softmax

* Derivative

.. literalinclude:: ./../nnlibs/commons/maths.py
    :pyobject: dsoftmax


metrics
~~~~~~~~~~~~~~~
models
~~~~~~~~~~~~~~~

.. literalinclude:: ./../nnlibs/commons/models.py
   :lines: 1-10

.. literalinclude:: ./../nnlibs/commons/models.py
    :pyobject: dataSet

.. literalinclude:: ./../nnlibs/commons/models.py
    :pyobject: runData

.. literalinclude:: ./../nnlibs/commons/models.py
    :pyobject: hPars

plot
~~~~~~~~~~~~~~~



schedule
~~~~~~~~~~~~~~~

.. include:: ./../nnlibs/commons/schedule.py
   :literal:
