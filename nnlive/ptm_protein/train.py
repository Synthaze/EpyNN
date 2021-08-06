# EpyNN/nnlive/ptm_protein/train.py
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
from nnlibs.network.models import EpyNN
from nnlibs.embedding.models import Embedding
from nnlibs.rnn.models import RNN
from nnlibs.lstm.models import LSTM
from nnlibs.gru.models import GRU
from nnlibs.flatten.models import Flatten
from nnlibs.dense.models import Dense
from prepare_dataset import labeled_dataset

from settings import (
    dataset as se_dataset,
    config as se_config,
    hPars as se_hPars,
)


from nnlibs.commons.maths import softmax
from nnlibs.dropout.models import Dropout
################################## HEADERS ################################
np.set_printoptions(precision=3, threshold=sys.maxsize)

np.seterr(all='warn')

SEED = 1
np.random.seed(SEED)

configure_directory(se_config)


################################## DATASETS ################################
dataset = labeled_dataset(se_dataset)
#dataset = read_dataset()


################################ BUILD MODEL ###############################
settings = [se_dataset, se_config, se_hPars]

embedding = Embedding(dataset, se_dataset, encode=True)

name = 'Embedding_Flatten_Dense-2-Softmax' # (1)
layers = [embedding, Flatten(), Dense(16, relu), Dense(2, softmax)]

name = 'Embedding_Flatten_RNN-11-Softmax_Dense-2-Softmax' # (5)
layers = [embedding, RNN(22), Flatten(), Dense(2, softmax)]


model = EpyNN(layers=layers,settings=settings, seed=SEED)


################################ TRAIN MODEL ################################
model.train()

model.plot()
