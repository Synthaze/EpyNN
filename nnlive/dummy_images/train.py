#EpyNN/nnlive/mnist_database/train.py
################################## IMPORTS ################################
# Set default environment and settings
from nnlibs.initialize import *
# EpyNN meta-model for neural networks
from nnlibs.meta.models import EpyNN
# Embedding layer for input data
from nnlibs.embedding.models import Embedding
# Import models specific to layer architectures
from nnlibs.conv.models import Convolution
from nnlibs.flatten.models import Flatten
from nnlibs.pool.models import Pooling
from nnlibs.dense.models import Dense
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
#dataset = pd.read_dataset()


################################ BUILD MODEL ###############################
embedding = Embedding(dataset,se.dataset)
convolution = Convolution(32,2)
pooling = Pooling(3,3)

name = 'Embedding_Flatten_Dense_Dense-2-Softmax'
layers = [embedding,Flatten(),Dense(64,cm.relu),Dense()]

# name = 'Embedding_Convolution_Pooling_Flatten_Dense_Dense-2-Softmax'
# layers = [embedding,convolution,pooling,Flatten(),Dense(64,cm.relu),Dense()]


model = EpyNN(name=name,layers=layers,settings=[se.dataset,se.config,se.hPars])


################################ TRAIN MODEL ################################
model.train()

model.plot()


################################# USE MODEL #################################
model = cl.read_model()

unlabeled_dataset = pd.prepare_unlabeled()

X = model.embedding_unlabeled(unlabeled_dataset)
#

A = model.predict(X)
#

P = np.argmax(A,axis=1)
#
