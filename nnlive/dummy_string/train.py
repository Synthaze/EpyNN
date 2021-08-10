# EpyNN/nnlive/dummy_string/train.py
# Standard library imports
import random

# Related third party imports
import numpy as np

# Local application/library specific imports
import nnlibs.initialize
from nnlibs.commons.maths import relu, softmax
from nnlibs.commons.library import (
    configure_directory,
    read_model,
)
from nnlibs.network.models import EpyNN
from nnlibs.embedding.models import Embedding
from nnlibs.flatten.models import Flatten
from nnlibs.rnn.models import RNN
from nnlibs.gru.models import GRU
from nnlibs.dense.models import Dense
from prepare_dataset import prepare_dataset
from settings import se_hPars


########################## CONFIGURE ##########################
random.seed(1)
np.random.seed(1)

np.set_printoptions(threshold=10)

np.seterr(all='warn')

configure_directory()


############################ DATASET ##########################
X_features, Y_label = prepare_dataset(N_SAMPLES=128)

embedding = Embedding(X_data=X_features,
                      Y_data=Y_label,
                      X_encode=True,
                      Y_encode=True,
                      relative_size=(2, 1, 0))


####################### BUILD AND TRAIN MODEL #################
# name = 'Flatten_Dense-2-softmax'
#
# flatten = Flatten()
#
# dense = Dense(2, softmax)
#
# layers = [embedding, flatten, dense]
#
# model = EpyNN(layers=layers, name=name)
#
# model.initialize(loss='MSE', seed=1)
#
# model.train(epochs=100)


# name = 'rnn-12_Flatten_Dense-2-softmax'
#
# se_hPars['learning_rate'] = 0.1
#
# rnn = RNN(12)
#
# dense = Dense(2, softmax)
#
# layers = [embedding, rnn, dense]
#
# model = EpyNN(layers=layers, name=name)
#
# model.initialize(loss='MSE', seed=1, se_hPars=se_hPars)
#
# model.train(epochs=10)
#
# model.plot(path=False)


# name = 'rnn-12_rnn-12_Flatten_Dense-2-softmax'
#
# rnn = RNN(12, sequences=True)
#
# rnn_bis = RNN(12)
#
# dense = Dense(2, softmax)
#
# layers = [embedding, rnn, rnn_bis, dense]
#
# model = EpyNN(layers=layers, name=name)
#
# model.initialize(loss='MSE', seed=1, se_hPars=se_hPars)
#
# model.train(epochs=10)
#
# model.plot(path=False)


# name = 'rnn-12_gru-12_Flatten_Dense-4-relu_Dense-2-softmax'
#
# rnn = RNN(12, sequences=True)
#
# gru = GRU(12, sequences=True)
#
# flatten = Flatten()
#
# hidden_dense = Dense(4, relu)
#
# dense = Dense(2, softmax)
#
# layers = [embedding, rnn, gru, flatten, hidden_dense, dense]
#
# model = EpyNN(layers=layers, name=name)
#
# model.initialize(loss='MSE', seed=1, se_hPars=se_hPars)
#
# model.train(epochs=100)
#
# model.plot(path=False)
