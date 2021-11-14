# EpyNN/epynnlive/dummy_string/train.py
# Standard library imports
import random

# Related third party imports
import numpy as np

# Local application/library specific imports
import epynn.initialize
from epynn.commons.io import one_hot_decode_sequence
from epynn.commons.maths import relu, softmax
from epynn.commons.library import (
    configure_directory,
    read_model,
)
from epynn.network.models import EpyNN
from epynn.embedding.models import Embedding
from epynn.flatten.models import Flatten
from epynn.rnn.models import RNN
from epynn.gru.models import GRU
from epynn.lstm.models import LSTM
from epynn.dense.models import Dense
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
embedding = Embedding(X_data=X_features,
                      Y_data=Y_label,
                      X_encode=True,
                      Y_encode=True,
                      relative_size=(2, 1, 0))

### Feed-Forward

# Model
name = 'Flatten_Dense-2-softmax'

se_hPars['learning_rate'] = 0.001

flatten = Flatten()

dense = Dense(2, softmax)

layers = [embedding, flatten, dense]

model = EpyNN(layers=layers, name=name)

model.initialize(loss='BCE', seed=1, se_hPars=se_hPars.copy())

model.train(epochs=50, init_logs=False)

model.plot(path=False)


### Recurrent

# Model
name = 'RNN-1_Dense-2-softmax'

se_hPars['learning_rate'] = 0.001

rnn = RNN(1)

dense = Dense(2, softmax)

layers = [embedding, rnn, dense]

model = EpyNN(layers=layers, name=name)

model.initialize(loss='BCE', seed=1, se_hPars=se_hPars.copy())

model.train(epochs=50, init_logs=False)

model.plot(path=False)


# Model
name = 'LSTM-1_Dense-2-softmax'

se_hPars['learning_rate'] = 0.005

lstm = LSTM(1)

dense = Dense(2, softmax)

layers = [embedding, lstm, dense]

model = EpyNN(layers=layers, name=name)

model.initialize(loss='BCE', seed=1, se_hPars=se_hPars.copy())

model.train(epochs=50, init_logs=False)

model.plot(path=False)


# Model
name = 'GRU-1_Dense-2-softmax'

se_hPars['learning_rate'] = 0.005

gru = GRU(1)

flatten = Flatten()

dense = Dense(2, softmax)

layers = [embedding, gru, dense]

model = EpyNN(layers=layers, name=name)

model.initialize(loss='BCE', seed=1, se_hPars=se_hPars.copy())

model.train(epochs=50, init_logs=False)

model.plot(path=False)


### Write/read model

model.write()
# model.write(path=/your/custom/path)

model = read_model()
# model = read_model(path=/your/custom/path)


### Predict

X_features, _ = prepare_dataset(N_SAMPLES=10)

dset = model.predict(X_features, X_encode=True)

for n, pred, probs in zip(dset.ids, dset.P, dset.A):
    print(n, pred, probs)
