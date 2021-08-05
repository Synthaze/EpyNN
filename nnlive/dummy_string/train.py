# EpyNN/nnlive/dummy_string/train.py
# Standard library imports
import random
import sys

# Related third party imports
import numpy as np

# Local application/library specific imports
import nnlibs.initialize
from nnlibs.commons.io import one_hot_decode_sequence
from nnlibs.commons.logs import pretty_json
from nnlibs.commons.library import (
    configure_directory,
    read_dataset,
    read_model,
)
from nnlibs.commons.maths import relu
from nnlibs.meta.models import EpyNN
from nnlibs.embedding.models import Embedding
from nnlibs.rnn.models import RNN
# from nnlibs.lstm.models import LSTM
# from nnlibs.gru.models import GRU
from nnlibs.flatten.models import Flatten
from nnlibs.dense.models import Dense
from prepare_dataset import (
    labeled_dataset,
    unlabeled_dataset,
)
from settings import (
    dataset as se_dataset,
    config as se_config,
    hPars as se_hPars,
)


############################ HEADERS ##########################
random.seed(1)

np.set_printoptions(precision=3,threshold=sys.maxsize)

np.seterr(all='warn')

configure_directory(se_config=se_config)

settings = [se_dataset, se_config, se_hPars]

############################ DATASET ##########################
dataset = labeled_dataset(se_dataset)

embedding = Embedding(dataset, relative_size=(2, 1, 0), X_encode=True, Y_encode=True)


############################ MODEL ############################
# Many-to-one RNN network
#bin_rnn = RNN(12, binary=True)
name = 'Embedding_RNN-12-Binary'
#model = EpyNN(layers=[embedding, bin_rnn], settings=settings, seed=1, name=name)

# Many-to-many RNN network
rnn = RNN(12)
flatten = Flatten()
dense = Dense()
#name = 'Embedding_RNN-12_Flatten_Dense'
model = EpyNN(layers=[embedding, rnn, flatten, dense], settings=settings, seed=1, name=name)


########################### TRAINING ###########################
model.train()
model.plot()
model.evaluate(write=True)


######################### PREDICTION ###########################
model = read_model()

unlabeled_more = unlabeled_dataset(N_SAMPLES=10)

dset = model.predict(unlabeled_more, X_encode=True)

dset.P = np.argmax(dset.A, axis=1)

for i in range(len(unlabeled_more)):
    print(unlabeled_more[i], dset.A[i], dset.P[i])
