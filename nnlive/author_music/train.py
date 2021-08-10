# EpyNN/nnlive/music_author/train.py
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
# from nnlibs.lstm.models import LSTM
# from nnlibs.gru.models import GRU
from nnlibs.flatten.models import Flatten
# from nnlibs.dropout.models import Dropout
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

embedding = Embedding(X_data=X_features,
                      Y_data=Y_label,
                      X_encode=True,
                      Y_encode=True,
                      batch_size=32,
                      relative_size=(2, 1, 0))

# flatten = Flatten()
#
# hidden_dense = Dense(64, relu)
#
# dense = Dense(2, softmax)
#
# layers = [embedding, flatten, hidden_dense, dense]
#
# model = EpyNN(layers=layers, name='None')
#
# model.train(epochs=100)
#
se_hPars['learning_rate'] = 0.01
se_hPars['schedule'] = 'exp_decay'
se_hPars['decay_k'] = 0.01

rnn = RNN(220, sequences=False)

flatten = Flatten()

dense = Dense(2, softmax)

layers = [embedding, rnn, flatten, dense]

model = EpyNN(layers=layers, name='None')

model.initialize(loss='BCE', se_hPars=se_hPars)

model.train(epochs=100)
