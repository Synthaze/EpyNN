# EpyNN/nnlive/dummy_boolean/train.py
# Standard library imports
import random
# Related third party imports
import numpy as np
# Local application/library specific imports
from nnlibs.initialize import *
from nnlibs.meta.models import EpyNN
from nnlibs.embedding.models import Embedding
from nnlibs.dense.models import Dense
import nnlibs.commons.library as cl
import nnlibs.commons.maths as cm
import prepare_dataset as pd
import settings as se




################################## HEADERS ################################

random.seed(1)

np.set_printoptions(precision=3, threshold=sys.maxsize)

np.seterr(all='warn')

cl.init_dir(se.config)

# DOCS_HEADERS
################################## DATASETS ################################
dataset = pd.prepare_dataset(se.dataset) # See "Data preparation, structure and shape"
#dataset = cl.read_dataset()

################################ BUILD MODEL ###############################
name = 'Embedding_Dense-2-Softmax' # (1)
layers = [Embedding(dataset, se.dataset), Dense(2)] # Dense() same as Dense(nodes=2,activate=cm.softmax)

#name = 'Dense-2-Sigmoid' # (2)
#layers = [Dense(nodes=2,activate=cm.sigmoid)]

#name = 'Dense-8-ReLU_Dense-2-Softmax' # (3)
#layers = [Dense(8,activate=cm.relu),Dense(2)]

model = EpyNN(layers=layers, settings=[se.dataset, se.config, se.hPars], seed=1, name=name)


################################ TRAIN MODEL ################################
model.train()

model.plot()


################################# USE MODEL #################################
model = cl.read_model()

unlabeled_dataset = pd.prepare_unlabeled(N_SAMPLES=1)

X = model.embedding_unlabeled(unlabeled_dataset)
# [[ True  True  True False  True False False False  True False  True]]

A = model.predict(X)
# [[0.714 0.286]]

P = np.argmax(A, axis=1)
# [0]
