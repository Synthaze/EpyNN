# EpyNN/nnlive/ptm_protein/train.py
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
from nnlibs.rnn.models import RNN
from nnlibs.lstm.models import LSTM
from nnlibs.gru.models import GRU
from nnlibs.flatten.models import Flatten
from nnlibs.dropout.models import Dropout
from nnlibs.dense.models import Dense
from prepare_dataset import prepare_dataset
from settings import se_hPars


########################## CONFIGURE ##########################
random.seed(1)

np.set_printoptions(threshold=10)

np.seterr(all='warn')
np.seterr(under='ignore')

############################ DATASET ##########################
X_features, Y_label = prepare_dataset(N_SAMPLES=1280)


####################### BUILD AND TRAIN MODEL #################
embedding = Embedding(X_data=X_features,
                      Y_data=Y_label,
                      X_encode=True,
                      Y_encode=True,
                      batch_size=32,
                      relative_size=(2, 1, 0))




### Feed-Forward

# name = 'Flatten_Dropout-08_Dense-64-relu_Dropout-07_Dense-2-softmax'
#
# se_hPars['learning_rate'] = 0.005
#
# flatten = Flatten()
#
# dropout1 = Dropout(keep_prob=0.8)
#
# hidden_dense = Dense(64, relu)
#
# dropout2 = Dropout(keep_prob=0.7)
#
# dense = Dense(2, softmax)
#
# layers = [embedding, flatten, dropout1, hidden_dense, dropout2, dense]
#
# model = EpyNN(layers=layers, name=name)
#
# model.initialize(loss='MSE', seed=1, se_hPars=se_hPars.copy())
#
# model.train(epochs=100, init_logs=False)


### Recurrent

# name = 'LSTM_Dense-2-softmax'
#
# se_hPars['learning_rate'] = 0.1
# se_hPars['softmax_temperature'] = 5
#
# lstm = LSTM(21)
#
# dense = Dense(2, softmax)
#
# layers = [embedding, lstm, dense]
#
# model = EpyNN(layers=layers, name=name)
#
# model.initialize(loss='BCE', seed=1, se_hPars=se_hPars.copy())
#
# model.train(epochs=100, init_logs=False)


# name = 'LSTM-21-Seq_Flatten_Dense-64-relu_Dropout-07_Dense-2-softmax'
#
# se_hPars['learning_rate'] = 0.01
# se_hPars['softmax_temperature'] = 5
#
# lstm = LSTM(21, sequences=True)
#
# flatten = Flatten()
#
# hidden_dense = Dense(64, relu)
#
# dropout2 = Dropout(keep_prob=0.7)
#
# dense = Dense(2, softmax)
#
# layers = [embedding, lstm, flatten, hidden_dense, dropout2, dense]
#
# model = EpyNN(layers=layers, name=name)
#
# model.initialize(loss='BCE', seed=1, se_hPars=se_hPars.copy())
#
# model.train(epochs=100, init_logs=False)
