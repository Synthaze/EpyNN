# EpyNN

![](https://img.shields.io/github/languages/top/synthaze/epynn) ![](https://img.shields.io/github/license/synthaze/epynn) ![](https://img.shields.io/github/last-commit/synthaze/epynn)

![](https://img.shields.io/github/stars/synthaze/epynn?style=social) ![](https://img.shields.io/twitter/follow/epynn_synthaze?label=Follow&style=social)

**EpyNN is written in pure Python/NumPy.**

If you use EpyNN in academia, please cite:

Malard F., Danner L., Rouzies E., Meyer J. G., Lescop E., Olivier-Van Stichelen S. [**EpyNN: Educational python for Neural Networks**](https://www.softxjournal.com/article/S2352-7110(22)00090-5/fulltext), *SoftwareX* 19 (2022).

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
git clone https://github.com/synthaze/EpyNN

# Change directory to EpyNN
cd EpyNN

# Install EpyNN dependencies
pip3 install -r requirements.txt

# Export EpyNN path in $PYTHONPATH for current session
export PYTHONPATH=$PYTHONPATH:$PWD

# Alternatively, not recommended
# pip3 install EpyNN
# epynn
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
git clone https://github.com/synthaze/EpyNN

# Change directory to EpyNN
chdir EpyNN

# Install EpyNN dependencies
pip3 install -r requirements.txt

# Show full path of EpyNN directory
echo %cd%

# Alternatively, not recommended
# pip3 install EpyNN
# epynn
```

Copy the full path of EpyNN directory, then go to:
``Control Panel > System > Advanced > Environment variable``

If you already have ``PYTHONPATH`` in the ``User variables`` section, select it and click ``Edit``, otherwise click ``New`` to add it.

Paste the full path of EpyNN directory in the input field, keep in mind that paths in ``PYTHONPATH`` should be comma-separated.

ANSI coloring schemes do work on native Windows10 and later. For prior Windows versions, users should configure their environment to work with ANSI coloring schemes for optimal experience.

## Current release

### 1.2 Publication release

* Minor revisions for peer-review process.

See [CHANGELOG.md](CHANGELOG.md) for past releases.



## Project tree

**epynn**
 * [convolution](epynn/convolution)
   * [backward.py](epynn/convolution/backward.py)
   * [forward.py](epynn/convolution/forward.py)
   * [models.py](epynn/convolution/models.py)
   * [parameters.py](epynn/convolution/parameters.py)
 * [dense](epynn/dense)
   * [backward.py](epynn/dense/backward.py)
   * [forward.py](epynn/dense/forward.py)
   * [models.py](epynn/dense/models.py)
   * [parameters.py](epynn/dense/parameters.py)
 * [dropout](epynn/dropout)
   * [backward.py](epynn/dropout/backward.py)
   * [forward.py](epynn/dropout/forward.py)
   * [models.py](epynn/dropout/models.py)
   * [parameters.py](epynn/dropout/parameters.py)
 * [embedding](epynn/embedding)
   * [backward.py](epynn/embedding/backward.py)
   * [dataset.py](epynn/embedding/dataset.py)
   * [forward.py](epynn/embedding/forward.py)
   * [models.py](epynn/embedding/models.py)
   * [parameters.py](epynn/embedding/parameters.py)
 * [flatten](epynn/flatten)
   * [backward.py](epynn/flatten/backward.py)
   * [forward.py](epynn/flatten/forward.py)
   * [models.py](epynn/flatten/models.py)
   * [parameters.py](epynn/flatten/parameters.py)
 * [gru](epynn/gru)
   * [backward.py](epynn/gru/backward.py)
   * [forward.py](epynn/gru/forward.py)
   * [models.py](epynn/gru/models.py)
   * [parameters.py](epynn/gru/parameters.py)
 * [lstm](epynn/lstm)
   * [backward.py](epynn/lstm/backward.py)
   * [forward.py](epynn/lstm/forward.py)
   * [models.py](epynn/lstm/models.py)
   * [parameters.py](epynn/lstm/parameters.py)
 * [pooling](epynn/pooling)
   * [backward.py](epynn/pooling/backward.py)
   * [forward.py](epynn/pooling/forward.py)
   * [models.py](epynn/pooling/models.py)
   * [parameters.py](epynn/pooling/parameters.py)
 * [rnn](epynn/rnn)
   * [backward.py](epynn/rnn/backward.py)
   * [forward.py](epynn/rnn/forward.py)
   * [models.py](epynn/rnn/models.py)
   * [parameters.py](epynn/rnn/parameters.py)
 * [template](epynn/template)
     * [backward.py](epynn/template/backward.py)
     * [forward.py](epynn/template/forward.py)
     * [models.py](epynn/template/models.py)
     * [parameters.py](epynn/template/parameters.py)
 * [network](epynn/network)
   * [backward.py](epynn/network/backward.py)
   * [evaluate.py](epynn/network/evaluate.py)
   * [forward.py](epynn/network/forward.py)
   * [hyperparameters.py](epynn/network/hyperparameters.py)
   * [initialize.py](epynn/network/initialize.py)
   * [models.py](epynn/network/models.py)
   * [report.py](epynn/network/report.py)
   * [training.py](epynn/network/training.py)
 * [commons](epynn/commons)
   * [io.py](epynn/commons/io.py)
   * [library.py](epynn/commons/library.py)
   * [logs.py](epynn/commons/logs.py)
   * [loss.py](epynn/commons/loss.py)
   * [maths.py](epynn/commons/maths.py)
   * [metrics.py](epynn/commons/metrics.py)
   * [models.py](epynn/commons/models.py)
   * [plot.py](epynn/commons/plot.py)
   * [schedule.py](epynn/commons/schedule.py)

**epynnlive**
 * [author_music](epynnlive/author_music)
 * [captcha_mnist](epynnlive/captcha_mnist)
 * [dummy_boolean](epynnlive/dummy_boolean)
 * [dummy_image](epynnlive/dummy_image)
 * [dummy_string](epynnlive/dummy_string)
 * [dummy_time](epynnlive/dummy_time)
 * [ptm_protein](epynnlive/ptm_protein)
