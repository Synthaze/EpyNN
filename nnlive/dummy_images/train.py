#EpyNN/nnlive/mnist_database/train.py
# Set environment and import default settings
from nnlibs.initialize import *
# Import common models
from nnlibs.commons.models import runData, hPars
# Import EpyNN meta-model to build neural networks
from nnlibs.meta.models import EpyNN
# Import models specific to layer architectures
from nnlibs.conv.models import Convolution
from nnlibs.dense.models import Dense
from nnlibs.dropout.models import Dropout
from nnlibs.flatten.models import Flatten
from nnlibs.gru.models import GRU
from nnlibs.lstm.models import LSTM
from nnlibs.pool.models import Pooling
from nnlibs.rnn.models import RNN
# Import utils
import nnlibs.commons.library as cl
import nnlibs.commons.maths as cm
# Import data-specific routine to prepare sets
import sets_prepare as sp
# Import local EpyNN settings
import settings as se

import numpy as np


np.set_printoptions(precision=3,threshold=sys.maxsize)

np.seterr(all='warn')

cm.global_seed(1)

cl.init_dir(se.config)

runData = runData(se.config)

hPars = hPars(se.hPars)

dsets = sp.sets_prepare(runData)

name = 'Convolution_Pooling_Flatten_Dense'
#name = 'Convolution_Flatten_Dense_Dense'
layers = [Convolution(32,2),Pooling(3,3),Flatten(),Dense(48),Dense(2)]
#layers = [Flatten(),Dense(48),Dense(2)]

model = EpyNN(name=name,layers=layers,hPars=hPars)

model.train(dsets,hPars,runData)

model.plot(hPars,runData)
