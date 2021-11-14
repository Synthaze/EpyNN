# EpyNN/epynnlive/ptm_protein/train.py
# Standard library imports
import random

# Related third party imports
import numpy as np

# Local application/library specific imports
import epynn.initialize
from epynn.commons.maths import relu, softmax
from epynn.commons.library import ( 
    configure_directory,
    read_model,
    write_model,
)
from epynn.network.models import EpyNN
from epynn.embedding.models import Embedding
from epynn.lstm.models import LSTM
from epynn.flatten.models import Flatten
from epynn.dropout.models import Dropout
from epynn.dense.models import Dense
from prepare_dataset import (
    prepare_dataset,
    download_sequences,
)
from settings import se_hPars


########################## CONFIGURE ##########################
random.seed(1)

np.set_printoptions(threshold=10)

np.seterr(all='warn')
np.seterr(under='ignore')

configure_directory()

############################ DATASET ##########################
download_sequences()

X_features, Y_label = prepare_dataset(N_SAMPLES=1280)

####################### BUILD AND TRAIN MODEL #################
embedding = Embedding(X_data=X_features,
                      Y_data=Y_label,
                      X_encode=True,
                      Y_encode=True,
                      batch_size=32,
                      relative_size=(2, 1, 0))


### Feed-Forward

# Model
name = 'Flatten_Dropout-02_Dense-64-relu_Dropout-03_Dense-2-softmax'

se_hPars['learning_rate'] = 0.001

flatten = Flatten()

dropout1 = Dropout(drop_prob=0.2)

hidden_dense = Dense(64, relu)

dropout2 = Dropout(drop_prob=0.3)

dense = Dense(2, softmax)

layers = [embedding, flatten, dropout1, hidden_dense, dropout2, dense]

model = EpyNN(layers=layers, name=name)

model.initialize(loss='MSE', seed=1, se_hPars=se_hPars.copy())

model.train(epochs=100, init_logs=False)

model.plot(path=False)


### Recurrent

# Model
name = 'LSTM-8_Dense-2-softmax'

se_hPars['learning_rate'] = 0.1
se_hPars['softmax_temperature'] = 5

lstm = LSTM(8)

dense = Dense(2, softmax)

layers = [embedding, lstm, dense]

model = EpyNN(layers=layers, name=name)

model.initialize(loss='MSE', seed=1, se_hPars=se_hPars.copy())

model.train(epochs=20, init_logs=False)

# Model
name = 'LSTM-8-Seq_Flatten_Dense-2-softmax'

se_hPars['learning_rate'] = 0.1
se_hPars['softmax_temperature'] = 5

lstm = LSTM(8, sequences=True)

flatten = Flatten()

dense = Dense(2, softmax)

layers = [embedding, lstm, flatten, dense]

model = EpyNN(layers=layers, name=name)

model.initialize(loss='MSE', seed=1, se_hPars=se_hPars.copy())

model.train(epochs=20, init_logs=False)

model.plot(path=False)

# Model
name = 'LSTM-8-Seq_Flatten_Dropout-05_Dense-64-relu_Dropout-05_Dense-2-softmax'

se_hPars['learning_rate'] = 0.1
se_hPars['softmax_temperature'] = 5

layers = [
    embedding,
    LSTM(8, sequences=True),
    Flatten(),
    Dropout(drop_prob=0.5),
    Dense(64, relu),
    Dropout(drop_prob=0.5),
    Dense(2, softmax),
]

model = EpyNN(layers=layers, name=name)

model.initialize(loss='MSE', seed=1, se_hPars=se_hPars.copy(), end='\r')

model.train(epochs=20, init_logs=False)

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
