#EpyNN/nnlive/dummy_boolean/train.py
################################## IMPORTS ################################
# Set environment and import default settings
from nnlibs.initialize import *
# Import EpyNN meta-model to build neural networks
from nnlibs.meta.models import EpyNN
#
from nnlibs.embedding.models import Embedding
# VARIABLE - Import models specific to layer architectures
from nnlibs.flatten.models import Flatten
from nnlibs.dense.models import Dense
from nnlibs.lstm.models import LSTM
from nnlibs.rnn.models import RNN
from nnlibs.gru.models import GRU
# Import utils
import nnlibs.commons.library as cl
import nnlibs.commons.maths as cm
# Import data-specific routine to prepare sets
import prepare_dataset as ps
# Import local EpyNN settings
import settings as se

import numpy as np


################################## HEADERS ################################
np.set_printoptions(precision=3,threshold=sys.maxsize)

np.seterr(all='warn')

cm.global_seed(1)

cl.init_dir(se.config)

# DOCS_HEADERS
################################## DATASETS ################################
dataset = ps.prepare_dataset(se.dataset) # See "Data preparation, structure and shape"
#dataset = ps.read_dataset()


################################ BUILD MODEL ###############################
embedding = Embedding(dataset,se.dataset,encode=True)

name = 'Embedding_Flatten_Dense-2-Softmax' # (1)
layers = [embedding,Flatten(),Dense()]

# name = 'RNN-11-bin-Softmax' # (2)
#layers = [embedding,RNN(hidden_size=11,binary=True)]

# name = 'GRU-11-bin-Softmax' # (3)
# layers = [embedding,GRU(11,binary=True)]

# name = 'LSTM-11-bin-Softmax' # (4)
# layers = [embedding,LSTM(11,binary=True)]

#name = 'Embedding_Flatten_RNN-11-Softmax_Dense-2-Softmax' # (5)
#layers = [embedding,RNN(11),Flatten(),Dense(48,cm.relu),Dense()]


model = EpyNN(name=name,layers=layers,settings=[se.dataset,se.config,se.hPars])


################################ TRAIN MODEL ################################
model.train()

model.plot()


################################# USE MODEL #################################
