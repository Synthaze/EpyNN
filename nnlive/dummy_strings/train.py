#EpyNN/nnlive/dummy_strings/train.py
################################## IMPORTS ################################
# Set environment and import default settings
from nnlibs.initialize import *
# Import common models
from nnlibs.commons.models import runData, hPars
# Import EpyNN meta-model to build neural networks
from nnlibs.meta.models import EpyNN
# Import models specific to layer architectures
from nnlibs.dense.models import Dense
from nnlibs.dropout.models import Dropout
from nnlibs.flatten.models import Flatten
from nnlibs.gru.models import GRU
from nnlibs.lstm.models import LSTM
from nnlibs.rnn.models import RNN
# Import utils
import nnlibs.commons.library as cl
import nnlibs.commons.maths as cm
# Import data-specific routine to prepare sets
import sets_prepare as sp
# Import local EpyNN settings
import settings as se

import numpy as np


################################## HEADERS ################################
np.set_printoptions(precision=3,threshold=sys.maxsize)

np.seterr(all='warn')

cm.global_seed(1)

cl.init_dir(se.config)

runData = runData(se.config)

hPars = hPars(se.hPars)


################################## DATASETS ################################
dsets = sp.sets_prepare(runData)


################################ BUILD MODEL ###############################
name = 'Flatten_Dense-2-Softmax'
layers = [Flatten(),Dense(2)]
# name = 'Flatten_Dense-8-ReLU_Dense-2-Softmax'
# layers = [Flatten(),Dense(8,activate=cm.elu),Dense(2)]
#
# name = 'RNN-11-bin'
# layers = [RNN(11,runData,binary=True)]
# name = 'GRU-11-bin'
# layers = [RNN(11,runData,binary=True)]
# name = 'LSTM-11-bin'
# layers = [LSTM(11,runData,binary=True)]

# name = 'RNN-11_Flatten_Dense'
# layers = [RNN(11,runData),Flatten(),Dense(2)]


################################ TRAIN MODEL ################################
model = EpyNN(name=name,layers=layers,hPars=hPars)

model.train(dsets,hPars,runData)

model.plot(hPars,runData)


################################# USE MODEL #################################
