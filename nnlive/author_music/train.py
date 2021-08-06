#EpyNN/nnlive/music_author/train.py
################################## IMPORTS ################################
# Set default environment and settings
from nnlibs.initialize import *
# EpyNN network-model for neural networks
from nnlibs.network.models import EpyNN
# Embedding layer for input data
from nnlibs.embedding.models import Embedding
# Import models specific to layer architectures
from nnlibs.dropout.models import Dropout
from nnlibs.flatten.models import Flatten
from nnlibs.dense.models import Dense
from nnlibs.lstm.models import LSTM
from nnlibs.rnn.models import RNN
from nnlibs.gru.models import GRU
# Commons utils and maths
import nnlibs.commons.library as cl
import nnlibs.commons.maths as cm
# Routines for dataset preparation
import prepare_dataset as pd
# Local EpyNN settings
import settings as se
# Compute with NumPy
import numpy as np


################################## HEADERS ################################
np.set_printoptions(precision=3,threshold=sys.maxsize)

np.seterr(all='warn')

cm.global_seed(1)

cl.init_dir(se.config)
# DOCS_HEADERS
################################## DATASETS ################################
dataset = pd.prepare_dataset(se.dataset)
#dataset = cl.read_dataset()


################################ BUILD MODEL ###############################
embedding = Embedding(dataset,se.dataset,encode=True)

name = 'Embedding_Flatten_Dense_Dense-2-Softmax' # (1)
layers = [embedding,Flatten(),Dense(16,cm.relu),Dense()]

# name = 'LSTM-128-bin-Softmax' # (2)
# layers = [embedding,RNN(400,binary=True)]

# name = 'Embedding_LSTM-128_Flatten_Dense_Dense-2-Softmax' # (3)
# layers = [embedding,LSTM(128),Flatten(),Dense(16,cm.relu),Dense()]


model = EpyNN(name=name,layers=layers,settings=[se.dataset,se.config,se.hPars])


################################ TRAIN MODEL ################################
model.train()

model.plot()
