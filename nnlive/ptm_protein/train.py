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
X_features, Y_label = prepare_dataset(N_SAMPLES=12800)

embedding = Embedding(X_data=X_features,
                      Y_data=Y_label,
                      X_encode=True,
                      Y_encode=True,
                      batch_size=128,
                      relative_size=(2, 1, 0))


lstm = LSTM(21)

flatten = Flatten()

dense = Dense(2, softmax)

layers = [embedding, lstm, flatten, dense]

model = EpyNN(layers=layers, name='O-GcNAc_lstmNet')

se_hPars['learning_rate'] = 0.1
se_hPars['schedule'] = 'exp_decay'

model.initialize(loss='BCE', se_hPars=se_hPars, seed=1)

model.train(epochs=20)

model.plot()


# flatten = Flatten()
#
# hidden_dense1 = Dense(128, relu)
#
# hidden_dense2 = Dense(16, relu)
#
# dense = Dense(2, softmax)
#
# layers = [embedding, flatten, hidden_dense1, hidden_dense2, dense]
#
# model = EpyNN(layers=layers, name='O-GcNAc_DeepNet')
#
# se_hPars['learning_rate'] = 0.01
# se_hPars['schedule'] = 'steady'
#
# model.initialize(loss='BCE', se_hPars=se_hPars, seed=1)
#
# model.train(epochs=20)
#
# model.plot()
