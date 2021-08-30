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
from nnlibs.gru.models import GRU
from nnlibs.flatten.models import Flatten
from nnlibs.dropout.models import Dropout
from nnlibs.dense.models import Dense
from prepare_dataset import (
    prepare_dataset,
    download_music,
)
from settings import se_hPars


########################## CONFIGURE ##########################
random.seed(1)

np.set_printoptions(threshold=10)

np.seterr(all='warn')
np.seterr(under='ignore')


############################ DATASET ##########################
download_music()

X_features, Y_label = prepare_dataset(N_SAMPLES=256)


####################### BUILD AND TRAIN MODEL #################
embedding = Embedding(X_data=X_features,
                      Y_data=Y_label,
                      X_encode=True,
                      Y_encode=True,
                      batch_size=16,
                      relative_size=(2, 1, 0))


### Feed-forward

# Model
name = 'Flatten_Dense-64-relu_Dense-2-softmax'

# se_hPars['learning_rate'] = 0.00001
# se_hPars['learning_rate'] = 1
se_hPars['learning_rate'] = 0.01
se_hPars['softmax_temperature'] = 1

layers = [
    embedding,
    Flatten(),
    Dense(64, relu),
    Dropout(0.5),
    Dense(2, softmax),
]

model = EpyNN(layers=layers, name=name)

model.initialize(loss='MSE', seed=1, metrics=['accuracy', 'recall', 'precision'], se_hPars=se_hPars.copy())

model.train(epochs=5, init_logs=False)


### Recurrent

# Model
name = 'RNN-1-Seq_Flatten_Dense-64-relu_Dropout05_Dense-2-softmax'

se_hPars['learning_rate'] = 0.01
se_hPars['softmax_temperature'] = 1

layers = [
    embedding,
    RNN(1, sequences=True),
    Flatten(),
    Dense(64, relu),
    Dropout(0.5),
    Dense(2, softmax),
]

model = EpyNN(layers=layers, name=name)

model.initialize(loss='MSE', seed=1, metrics=['accuracy', 'recall', 'precision'], se_hPars=se_hPars.copy())

model.train(epochs=5, init_logs=False)


# Model
name = 'GRU-1-Seq_Flatten_Dense-64-relu_Dropout05_Dense-2-softmax'

se_hPars['learning_rate'] = 0.01
se_hPars['softmax_temperature'] = 1

layers = [
    embedding,
    GRU(1, sequences=True),
    Flatten(),
    Dense(64, relu),
    Dropout(0.5),
    Dense(2, softmax),
]

model = EpyNN(layers=layers, name=name)

model.initialize(loss='MSE', seed=1, metrics=['accuracy', 'recall', 'precision'], se_hPars=se_hPars.copy())

model.train(epochs=5, init_logs=False)


### Write/read model

model.write()

model = read_model()


### Predict

X_features, _ = prepare_dataset(N_SAMPLES=10)

dset = model.predict(X_features, X_encode=True)

for n, pred, probs in zip(dset.ids, dset.P, dset.A):
    print(n, pred, probs)
