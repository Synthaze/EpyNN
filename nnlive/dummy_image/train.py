# EpyNN/nnlive/dummy_image/train.py
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
from nnlibs.convolution.models import Convolution
from nnlibs.pooling.models import Pooling
from nnlibs.flatten.models import Flatten
from nnlibs.dropout.models import Dropout
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
X_features, Y_label = prepare_dataset(N_SAMPLES=7500)


####################### BUILD AND TRAIN MODEL #################

### Feed-Forward
# embedding = Embedding(X_data=X_features,
#                       Y_data=Y_label,
#                       X_scale=True,
#                       Y_encode=True,
#                       batch_size=32,
#                       relative_size=(2, 1, 0))


# name = 'Flatten_Dense-2-softmax'
#
# se_hPars['learning_rate'] = 0.01
#
# flatten = Flatten()
#
# dense = Dense(2, softmax)
#
# layers = [embedding, flatten, dense]
#
# model = EpyNN(layers=layers, name=name)
#
# model.initialize(loss='MSE', seed=1, se_hPars=se_hPars.copy())
# # model.initialize(loss='BCE', seed=1, se_hPars=se_hPars.copy())
#
# model.train(epochs=100)


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


### Convolutional Neural Network
embedding = Embedding(X_data=X_features,
                      Y_data=Y_label,
                      X_scale=True,
                      Y_encode=True,
                      batch_size=32,
                      relative_size=(2, 1, 0))


name = 'Convolution-32-2_Pooling-3-3-Max_Flatten_Dense-2-softmax'

se_hPars['learning_rate'] = 0.005
se_hPars['softmax_temperature'] = 5

convolution = Convolution(n_filters=2, f_width=5, activate=relu)

pooling = Pooling(pool_size=(5, 5), stride=5)

flatten = Flatten()

# hidden_dense = Dense(128, relu)
dense = Dense(2, softmax)

layers = [embedding, convolution, pooling, flatten, dense]
# layers = [embedding, convolution, pooling, flatten, hidden_dense, dense]

model = EpyNN(layers=layers, name=name)

model.initialize(loss='CCE', seed=1, se_hPars=se_hPars.copy())

model.train(epochs=100, init_logs=False)
