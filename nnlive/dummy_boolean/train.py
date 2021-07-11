#EpyNN/nnlive/dummy_boolean/train.py
################################## IMPORTS ################################
# Set environment and import default settings
from nnlibs.initialize import *
# Import EpyNN meta-model to build neural networks
from nnlibs.meta.models import EpyNN
#
from nnlibs.embedding.models import Embedding
# VARIABLE - Import models specific to layer architectures
from nnlibs.dense.models import Dense
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
name = 'Embedding_Dense-2-Softmax' # (1)
layers = [Embedding(dataset,se.dataset),Dense()] # Dense() same as Dense(nodes=2,activate=cm.softmax)

#name = 'Dense-2-Sigmoid' # (2)
#layers = [Dense(nodes=2,activate=cm.sigmoid)]

#name = 'Dense-8-ReLU_Dense-2-Softmax' # (3)
#layers = [Dense(8,activate=cm.relu),Dense(2)]

model = EpyNN(name=name,layers=layers,settings=[se.dataset,se.config,se.hPars])


################################ TRAIN MODEL ################################
model.train()

model.plot()


################################# USE MODEL #################################
