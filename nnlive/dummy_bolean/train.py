#EpyNN/nnlive/dummy_bolean/train.py
################################## IMPORTS ################################
# Set environment and import default settings
from nnlibs.initialize import *
# Import common models
from nnlibs.commons.models import runData, hPars
# Import EpyNN meta-model to build neural networks
from nnlibs.meta.models import EpyNN
# Import models specific to layer architectures
from nnlibs.dense.models import Dense
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
name = 'Dense-2-Softmax'
layers = [Dense(2)]
#name = 'Dense-2-Sigmoid'
#layers = [Dense(2,activate=cm.sigmoid)]
#name = 'Dense-8-ReLU_Dense-2-Softmax'
#layers = [Dense(8,activate=cm.relu),Dense(2)]

model = EpyNN(name=name,layers=layers,hPars=hPars)


################################ TRAIN MODEL ################################
model.train(dsets,hPars,runData)

model.plot(hPars,runData)


################################# USE MODEL #################################
