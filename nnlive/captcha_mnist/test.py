# EpyNN/nnlive/captcha_mnist/train.py
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
from nnlibs.commons.maths import relu, softmax, sigmoid, tanh
from nnlibs.meta.models import EpyNN
from nnlibs.embedding.models import Embedding
from nnlibs.convolution.models import Convolution
from nnlibs.pooling.models import Pooling
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


################################## HEADERS ################################
np.set_printoptions(precision=3, threshold=sys.maxsize)

np.seterr(all='warn')

SEED = 1
np.random.seed(SEED)
random.seed(1)
configure_directory(se_config)


################################## DATASETS ################################
dataset = labeled_dataset(se_dataset)
#dataset = read_dataset()


################################ BUILD MODEL ###############################
settings = [se_dataset, se_config, se_hPars]

embedding = Embedding(dataset, se_dataset, scale=True)

convolution = Convolution(3, 3)
pooling = Pooling(pool_size=(2, 2), stride=2)

layers = [embedding, convolution, pooling, Flatten(), Dense(10, softmax)]

model = EpyNN(layers=layers,settings=settings, seed=SEED)

model.initialize()

model.train(init=False)
################################ TRAIN MODEL ################################
