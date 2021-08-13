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
from nnlibs.lstm.models import LSTM
from nnlibs.dense.models import Dense
from prepare_dataset import prepare_dataset
from settings import se_hPars


########################## CONFIGURE ##########################
random.seed(1)

np.set_printoptions(threshold=10)

np.seterr(all='warn')

configure_directory()


############################ DATASET ##########################
X_features, Y_label = prepare_dataset(N_SAMPLES=480)


####################### BUILD AND TRAIN MODEL #################

### Feed-Forward
# embedding = Embedding(X_data=X_features,
#                       Y_data=Y_label,
#                       X_encode=True,
#                       Y_encode=True,
#                       batch_size=32,
#                       relative_size=(2, 1, 0))
#
#
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
#
# model.plot(path=False)
#
#
# name = 'Flatten_Dense-16-relu_Dense-2-softmax'
#
# flatten = Flatten()
#
# hidden_dense = Dense(16, relu)
#
# dense = Dense(2, softmax)
#
# layers = [embedding, flatten, hidden_dense, dense]
#
# model = EpyNN(layers=layers, name=name)
#
# model.initialize(loss='BCE', seed=1)
#
# model.train(epochs=100)
#
# model.plot(path=False)


### Recurrent
embedding = Embedding(X_data=X_features,
                      Y_data=Y_label,
                      X_encode=True,
                      Y_encode=True,
                      batch_size=None,
                      relative_size=(2, 1, 0))

name = 'RNN-12-Seq_Flatten_Dense-2-softmax'

se_hPars['learning_rate'] = 0.01
se_hPars['schedule'] = 'exp_decay'

# rnn = RNN(12, sequences=True)

rnn = RNN(12)

flatten = Flatten()

dense = Dense(2, softmax)

layers = [embedding, rnn, dense]

model = EpyNN(layers=layers, name=name)

model.initialize(loss='MSE', seed=1, se_hPars=se_hPars.copy())

model.train(epochs=200, init_logs=False)
# name = 'RNN-12-Seq_Flatten_Dense-2-softmax'
#
# se_hPars['learning_rate'] = 0.01
# se_hPars['schedule'] = 'exp_decay'
#
# rnn = RNN(12, sequences=True)
#
# flatten = Flatten()
#
# dense = Dense(2, softmax)
#
# layers = [embedding, rnn, flatten, dense]
#
# model = EpyNN(layers=layers, name=name)
#
# model.initialize(loss='MSE', seed=1, se_hPars=se_hPars.copy())
#
# model.train(epochs=200)
#
# model.plot(path=False)


# name = 'LSTM-12-Seq_Flatten_Dense-2-softmax'
#
# se_hPars['learning_rate'] = 0.005
# se_hPars['schedule'] = 'steady'
#
# lstm = LSTM(12, sequences=True)
#
# dense = Dense(2, softmax)
#
# layers = [embedding, lstm, flatten, dense]
#
# model = EpyNN(layers=layers, name=name)
#
# model.initialize(loss='MSE', seed=1, se_hPars=se_hPars.copy())
#
# model.train(epochs=200)
#
# model.plot(path=False)


# name = 'GRU-12-Seq_Flatten_Dense-2-softmax'
#
# se_hPars['learning_rate'] = 0.005
# se_hPars['schedule'] = 'steady'
#
# gru = GRU(12, sequences=True)
#
# dense = Dense(2, softmax)
#
# layers = [embedding, gru, flatten, dense]
#
# model = EpyNN(layers=layers, name=name)
#
# model.initialize(loss='MSE', seed=1, se_hPars=se_hPars.copy())
#
# model.train(epochs=200)
#
# model.plot(path=False)


# model.write()


########################## PREDICTION #########################
# model = read_model()
#
# X_features, _ = prepare_dataset(N_SAMPLES=10)
#
# dset = model.predict(X_features, X_encode=True)
#
# for id, pred, probs, features in zip(dset.ids, dset.P, dset.A, dset.X):
#     print(id, pred, probs)
