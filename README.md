# EpyNN

![](https://img.shields.io/github/languages/top/synthaze/epynn) ![](https://img.shields.io/github/license/synthaze/epynn) ![](https://img.shields.io/github/last-commit/synthaze/epynn)

![](https://img.shields.io/github/stars/synthaze/epynn?style=social) ![](https://img.shields.io/twitter/follow/epynn_synthaze?label=Follow&style=social)
<p align="center">
    <a href="https://twitter.com/intent/follow?screen_name=epynn_synthaze" target="\_blank">
        <img src="https://img.shields.io/twitter/follow/epynn_synthaze?label=Follow&style=social" alt="follow on Twitter">
    </a>
</p>
**EpyNN is written in pure Python/NumPy.**

If you use EpyNN in academia, please cite:

Malard F., Danner L., Rouzies E., Meyer J. G., Lescop E., Olivier-Van Stichelen S. **EpyNN: Educational python for Neural Networks**, 2021, *Submitted*.

## Documentation

Please visit https://epynn.net/ for extensive documentation.

### Purpose

EpyNN is intended for **teachers**, **students**, **scientists**, or more generally anyone with minimal skills in Python programming **who wish to understand** and build from basic implementations of Neural Network architectures.

Although EpyNN can be used for production, it is meant to be a library of **homogeneous architecture templates** and **practical examples** which is expected to save an important amount of time for people who wish to learn, teach or **develop from scratch**.

### Content

EpyNN features **scalable**, **minimalistic** and **homogeneous** implementations of major Neural Network architectures in **pure Python/Numpy** including:

* [Embedding layer (Input)](https://epynn.net/Embedding.html)
* [Fully connected layer (Dense)](https://epynn.net/Dense.html)
* [Recurrent Neural Network (RNN)](https://epynn.net/RNN.html)
* [Long Short-Term Memory (LSTM)](https://epynn.net/LSTM.html)
* [Gated Recurrent Unit (GRU)](https://epynn.net/GRU.html)
* [Convolution (CNN)](https://epynn.net/Convolution.html)
* [Pooling (CNN)](https://epynn.net/Pooling.html)
* [Dropout - Regularization](https://epynn.net/Dropout.html)
* [Flatten - Adapter](https://epynn.net/Flatten.html)

Model and function rules and definition:

* [Architecture Layers - Model](https://epynn.net/EpyNN_Model.html)
* [Neural Network - Model](https://epynn.net/Layer_Model.html)
* [Data - Model](https://epynn.net/Data_Model.html)
* [Activation - Functions](https://epynn.net/activation.html)
* [Loss - Functions](https://epynn.net/loss.html)

While not enhancing, extending or replacing EpyNN's documentation, series of live examples in Python and Jupyter notebook formats are offered online and within the archive, including:

* [Data preparation - Examples](https://epynn.net/data_examples.html)
* [Network training - Examples](https://epynn.net/run_examples.html)

### Reliability

EpyNN has been cross-validated against TensorFlow/Keras API and provides identical results for identical configurations in the limit of float64 precision.

Please see [Is EpyNN reliable?](https://epynn.net/index.html#is-epynn-reliable) for details and executable codes.

### Recommended install

* **Linux/MacOS**

```bash

# Use bash shell
bash

# Clone git repository
git clone https://github.com/Synthaze/EpyNN

# Alternatively, not recommended
# pip3 install EpyNN
# epynn

# Change directory to EpyNN
cd EpyNN

# Install EpyNN dependencies
pip3 install -r requirements.txt

# Export EpyNN path in $PYTHONPATH for current session
export PYTHONPATH=$PYTHONPATH:$PWD

```

**Linux:** Permanent export of EpyNN directory path in ```$PYTHONPATH```.

```bash
# Append export instruction to the end of .bashrc file
echo "export PYTHONPATH=$PYTHONPATH:$PWD" >> ~/.bashrc

# Source .bashrc to refresh $PYTHONPATH
source ~/.bashrc
```

**MacOS:** Permanent export of EpyNN directory path in ```$PYTHONPATH```.

```bash
# Append export instruction to the end of .bash_profile file
echo "export PYTHONPATH=$PYTHONPATH:$PWD" >> ~/.bash_profile

# Source .bash_profile to refresh $PYTHONPATH
source ~/.bash_profile
```

* **Windows**

```bash
# Clone git repository
git clone https://github.com/Synthaze/EpyNN

# Alternatively, not recommended
# pip3 install EpyNN
# epynn

# Change directory to EpyNN
chdir EpyNN

# Install EpyNN dependencies
pip3 install -r requirements.txt

# Show full path of EpyNN directory
echo %cd%
```

Copy the full path of EpyNN directory, then go to:
``Control Panel > System > Advanced > Environment variable``

If you already have ``PYTHONPATH`` in the ``User variables`` section, select it and click ``Edit``, otherwise click ``New`` to add it.

Paste the full path of EpyNN directory in the input field, keep in mind that paths in ``PYTHONPATH`` should be comma-separated.

ANSI coloring schemes do work on native Windows10 and later. For prior Windows versions, users should configure their environment to work with ANSI coloring schemes for optimal experience.

## Current release

### 1.0 - Initial release

* **nnlibs** contains API sources.
* **nnlive** contains live examples in Python and Jupyter notebook formats.
* https://epynn.net/ contains extensive documentation.

See [CHANGELOG.md](CHANGELOG.md) for past releases.


## Project tree

**nnlibs**
 * [convolution](nnlibs/convolution)
   * [models.py](nnlibs/convolution/models.py)
   * [forward.py](nnlibs/convolution/forward.py)
   * [backward.py](nnlibs/convolution/backward.py)
   * [parameters.py](nnlibs/convolution/parameters.py)
 * [dense](nnlibs/dense)
   * [models.py](nnlibs/dense/models.py)
   * [backward.py](nnlibs/dense/backward.py)
   * [forward.py](nnlibs/dense/forward.py)
   * [parameters.py](nnlibs/dense/parameters.py)
 * [dropout](nnlibs/dropout)
   * [models.py](nnlibs/dropout/models.py)
   * [parameters.py](nnlibs/dropout/parameters.py)
   * [backward.py](nnlibs/dropout/backward.py)
   * [forward.py](nnlibs/dropout/forward.py)
 * [embedding](nnlibs/embedding)
   * [forward.py](nnlibs/embedding/forward.py)
   * [parameters.py](nnlibs/embedding/parameters.py)
   * [backward.py](nnlibs/embedding/backward.py)
   * [dataset.py](nnlibs/embedding/dataset.py)
   * [models.py](nnlibs/embedding/models.py)
 * [flatten](nnlibs/flatten)
   * [models.py](nnlibs/flatten/models.py)
   * [parameters.py](nnlibs/flatten/parameters.py)
   * [forward.py](nnlibs/flatten/forward.py)
   * [backward.py](nnlibs/flatten/backward.py)
 * [gru](nnlibs/gru)
   * [models.py](nnlibs/gru/models.py)
   * [forward.py](nnlibs/gru/forward.py)
   * [parameters.py](nnlibs/gru/parameters.py)
   * [backward.py](nnlibs/gru/backward.py)
 * [lstm](nnlibs/lstm)
   * [models.py](nnlibs/lstm/models.py)
   * [forward.py](nnlibs/lstm/forward.py)
   * [parameters.py](nnlibs/lstm/parameters.py)
   * [backward.py](nnlibs/lstm/backward.py)
 * [pooling](nnlibs/pooling)
   * [models.py](nnlibs/pooling/models.py)
   * [forward.py](nnlibs/pooling/forward.py)
   * [parameters.py](nnlibs/pooling/parameters.py)
   * [backward.py](nnlibs/pooling/backward.py)
 * [rnn](nnlibs/rnn)
   * [models.py](nnlibs/rnn/models.py)
   * [forward.py](nnlibs/rnn/forward.py)
   * [backward.py](nnlibs/rnn/backward.py)
   * [parameters.py](nnlibs/rnn/parameters.py)
 * [template](nnlibs/template)
     * [forward.py](nnlibs/template/forward.py)
     * [models.py](nnlibs/template/models.py)
     * [parameters.py](nnlibs/template/parameters.py)
     * [backward.py](nnlibs/template/backward.py)
 * [network](nnlibs/network)
   * [forward.py](nnlibs/network/forward.py)
   * [backward.py](nnlibs/network/backward.py)
   * [hyperparameters.py](nnlibs/network/hyperparameters.py)
   * [training.py](nnlibs/network/training.py)
   * [initialize.py](nnlibs/network/initialize.py)
   * [evaluate.py](nnlibs/network/evaluate.py)
   * [models.py](nnlibs/network/models.py)
   * [report.py](nnlibs/network/report.py)
 * [commons](nnlibs/commons)
   * [schedule.py](nnlibs/commons/schedule.py)
   * [library.py](nnlibs/commons/library.py)
   * [metrics.py](nnlibs/commons/metrics.py)
   * [loss.py](nnlibs/commons/loss.py)
   * [plot.py](nnlibs/commons/plot.py)
   * [io.py](nnlibs/commons/io.py)
   * [models.py](nnlibs/commons/models.py)
   * [logs.py](nnlibs/commons/logs.py)
   * [maths.py](nnlibs/commons/maths.py)

**nnlive**
 * [author_music](nnlive/author_music)
 * [dummy_boolean](nnlive/dummy_boolean)
 * [dummy_string](nnlive/dummy_string)
 * [ptm_protein](nnlive/ptm_protein)
 * [captcha_mnist](nnlive/captcha_mnist)
 * [dummy_image](nnlive/dummy_image)
 * [dummy_time](nnlive/dummy_time)
