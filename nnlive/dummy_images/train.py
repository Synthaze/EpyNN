#EpyNN/nnlive/mnist_database/train.py
################################## IMPORTS ################################
# Set environment and import default settings
from nnlibs.initialize import *
# Import EpyNN meta-model to build neural networks
from nnlibs.meta.models import EpyNN
# Import layer to embedd sample input data
from nnlibs.embedding.models import Embedding
# Import models specific to layer architectures
from nnlibs.conv.models import Convolution
from nnlibs.flatten.models import Flatten
from nnlibs.pool.models import Pooling
from nnlibs.dense.models import Dense
# Import utils
import nnlibs.commons.library as cl
import nnlibs.commons.maths as cm
# Import data-specific routine to prepare sets
import prepare_dataset as pd
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
dataset = pd.prepare_dataset(se.dataset)
#dataset = pd.read_dataset()


################################ BUILD MODEL ###############################
embedding = Embedding(dataset,se.dataset)
convolution = Convolution(32,2)
pooling = Pooling(3,3)

name = 'Embedding_Convolution_Pooling_Flatten_Dense_Dense-2-Softmax'
layers = [embedding,convolution,pooling,Flatten(),Dense(64,cm.relu),Dense()]


model = EpyNN(name=name,layers=layers,settings=[se.dataset,se.config,se.hPars])


################################ TRAIN MODEL ################################
model.train()

model.plot()


################################# USE MODEL #################################
