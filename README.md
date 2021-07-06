# EpyNN - 0.0.1

![alt text](https://github.com/Synthaze/EpyNN/blob/main/docs/logo.png)

## Last comments from development

### Current developpement axis with completion score (subjective).

1. EpyNN meta-structure: 90%
    - Add predict-only mode in nnlive/
    - Thinking about symlinks in nnlive/

3. Code organization: 75%
    - Improvement in nnlibs/commons
    - Think about general use of independant pseudo-random number generators
    - Code comments

3. EpyNN monitoring: 80%
    - Implement modes for logs (+color compatibility)
    

**Although running smoothly, code performance in this new version is essentially untested**

**Next goal is technical validation against previous versions / other librairies**

### Perspectives

1. Documentation
    - Technical
    - Examples

2. Jupyter?

4. Paper redaction

## Installation

Temporary instructions:

```
git clone https://github.com/synthaze/EpyNN

cd EpyNN

# May be outdated
pip3 install -r requirements.txt 

bash
# For current session:
export PYTHONPATH=$PYTHONPATH:$PWD

# Add the line to .bashrc (Linux) or .bash_profile (MacOS) for permanent change
echo "export PYTHONPATH=$PYTHONPATH:$PWD" >> ~/.bashrc
```

## Features

1. Architectures
    - Feed-forward (Dense)
    - Recurrent (RNN, GRU, LSTM)
    - Convolutional (Convolution + Pooling)

2. Layers
    - Dense

    - RNN
    - GRU
    - LSTM

    - Convolution
    - Pooling

    - Dropout

    - Flatten

3. Customization
    - ...

## Applications

1. Classification tasks
    - Binary labels
    - Multi-labels (>2)

2. Sequence prediction
    - Analogous sequences
    - Sequences extension

3. Data shape
    - 2D: Feed-forward
    - 3D: Recurrent, Feed-forward
    - 4D: Convolutional, Recurrent, Feed-forward

## Examples

1. Dummy examples
    - Boleans-Label
    - Strings-Label

2. List examples
    - ...

3. Sequence examples
    - Peptide-PTM (text)
    - Music-Genre (time)

4. Image examples
    - Peptide-PTM (image representation)
    - Captcha-Number (image)


## Intended audience

## Credits

## Project Tree

* [EpyNN](./EpyNN)
    * [nnlibs](./EpyNN/nnlibs)
        * [dropout](./EpyNN/nnlibs/dropout)
          * [models.py](./EpyNN/nnlibs/dropout/models.py)
          * [parameters.py](./EpyNN/nnlibs/dropout/parameters.py)
          * [backward.py](./EpyNN/nnlibs/dropout/backward.py)
          * [forward.py](./EpyNN/nnlibs/dropout/forward.py)
        * [flatten](./EpyNN/nnlibs/flatten)
          * [backward.py](./EpyNN/nnlibs/flatten/backward.py)
          * [models.py](./EpyNN/nnlibs/flatten/models.py)
          * [forward.py](./EpyNN/nnlibs/flatten/forward.py)
        * [lstm](./EpyNN/nnlibs/lstm)
          * [parameters.py](./EpyNN/nnlibs/lstm/parameters.py)
          * [models.py](./EpyNN/nnlibs/lstm/models.py)
          * [forward.py](./EpyNN/nnlibs/lstm/forward.py)
          * [backward.py](./EpyNN/nnlibs/lstm/backward.py)
        * [commons](./EpyNN/nnlibs/commons)
          * [plot.py](./EpyNN/nnlibs/commons/plot.py)
          * [schedule.py](./EpyNN/nnlibs/commons/schedule.py)
          * [decorators.py](./EpyNN/nnlibs/commons/decorators.py)
          * [maths.py](./EpyNN/nnlibs/commons/maths.py)
          * [library.py](./EpyNN/nnlibs/commons/library.py)
          * [io.py](./EpyNN/nnlibs/commons/io.py)
          * [models.py](./EpyNN/nnlibs/commons/models.py)
          * [logs.py](./EpyNN/nnlibs/commons/logs.py)
          * [metrics.py](./EpyNN/nnlibs/commons/metrics.py)
        * [conv](./EpyNN/nnlibs/conv)
          * [models.py](./EpyNN/nnlibs/conv/models.py)
          * [forward.py](./EpyNN/nnlibs/conv/forward.py)
          * [parameters.py](./EpyNN/nnlibs/conv/parameters.py)
          * [backward.py](./EpyNN/nnlibs/conv/backward.py)
        * [dense](./EpyNN/nnlibs/dense)
          * [forward.py](./EpyNN/nnlibs/dense/forward.py)
          * [backward.py](./EpyNN/nnlibs/dense/backward.py)
          * [parameters.py](./EpyNN/nnlibs/dense/parameters.py)
          * [models.py](./EpyNN/nnlibs/dense/models.py)
        * [gru](./EpyNN/nnlibs/gru)
          * [parameters.py](./EpyNN/nnlibs/gru/parameters.py)
          * [backward.py](./EpyNN/nnlibs/gru/backward.py)
          * [forward.py](./EpyNN/nnlibs/gru/forward.py)
          * [models.py](./EpyNN/nnlibs/gru/models.py)
        * [meta](./EpyNN/nnlibs/meta)
          * [models.py](./EpyNN/nnlibs/meta/models.py)
          * [train.py](./EpyNN/nnlibs/meta/train.py)
          * [forward.py](./EpyNN/nnlibs/meta/forward.py)
          * [backward.py](./EpyNN/nnlibs/meta/backward.py)
          * [parameters.py](./EpyNN/nnlibs/meta/parameters.py)
        * [pool](./EpyNN/nnlibs/pool)
          * [parameters.py](./EpyNN/nnlibs/pool/parameters.py)
          * [models.py](./EpyNN/nnlibs/pool/models.py)
          * [forward.py](./EpyNN/nnlibs/pool/forward.py)
          * [backward.py](./EpyNN/nnlibs/pool/backward.py)
        * [rnn](./EpyNN/nnlibs/rnn)
            * [models.py](./EpyNN/nnlibs/rnn/models.py)
            * [backward.py](./EpyNN/nnlibs/rnn/backward.py)
            * [parameters.py](./EpyNN/nnlibs/rnn/parameters.py)
            * [forward.py](./EpyNN/nnlibs/rnn/forward.py)
        * [initialize.py](./EpyNN/nnlibs/initialize.py)
        * [settings.py](./EpyNN/nnlibs/settings.py)
    * [nnlive](./EpyNN/nnlive)
      * [dummy_bolean](./EpyNN/nnlive/dummy_bolean)
        * [sets](./EpyNN/nnlive/dummy_bolean/sets)
        * [sets_prepare.py](./EpyNN/nnlive/dummy_bolean/sets_prepare.py)
        * [settings.py](./EpyNN/nnlive/dummy_bolean/settings.py)
        * [models](./EpyNN/nnlive/dummy_bolean/models)
        * [train.py](./EpyNN/nnlive/dummy_bolean/train.py)
      * [dummy_strings](./EpyNN/nnlive/dummy_strings)
        * [sets](./EpyNN/nnlive/dummy_strings/sets)
        * [sets_prepare.py](./EpyNN/nnlive/dummy_strings/sets_prepare.py)
        * [train.py](./EpyNN/nnlive/dummy_strings/train.py)
        * [settings.py](./EpyNN/nnlive/dummy_strings/settings.py)
        * [models](./EpyNN/nnlive/dummy_strings/models)
      * [mnist_database](./EpyNN/nnlive/mnist_database)
        * [sets](./EpyNN/nnlive/mnist_database/sets)
        * [sets_prepare.py](./EpyNN/nnlive/mnist_database/sets_prepare.py)
        * [settings.py](./EpyNN/nnlive/mnist_database/settings.py)
        * [models](./EpyNN/nnlive/mnist_database/models)
        * [train.py](./EpyNN/nnlive/mnist_database/train.py)
        * [data](./EpyNN/nnlive/mnist_database/data)
        * [train-images.gz](./EpyNN/nnlive/mnist_database/data/train-labels.gz)
        * [train-labels.gz](./EpyNN/nnlive/mnist_database/data/train-labels.gz)
      * [ptm_prediction](./EpyNN/nnlive/ptm_prediction)
      * [sets](./EpyNN/nnlive/ptm_prediction/sets)
      * [data](./EpyNN/nnlive/ptm_prediction/data)
        * [21_negative.dat](./EpyNN/nnlive/ptm_prediction/data/21_negative.dat)
        * [21_positive.dat](./EpyNN/nnlive/ptm_prediction/data/21_positive.dat)
      * [sets_prepare.py](./EpyNN/nnlive/ptm_prediction/sets_prepare.py)
      * [models](./EpyNN/nnlive/ptm_prediction/models)
      * [train.py](./EpyNN/nnlive/ptm_prediction/train.py)
      * [settings.py](./EpyNN/nnlive/ptm_prediction/settings.py)
    * [requirements.txt](./EpyNN/requirements.txt)
    * [LICENSE](./EpyNN/LICENSE)
    * [setup.cfg](./EpyNN/setup.cfg)
    * [README.md](./EpyNN/README.md)

