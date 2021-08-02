# EpyNN/nnlive/dummy_boolean/train.py
# Standard library imports
import random
import sys

# Related third party imports
import numpy as np

# Local application/library specific imports
import nnlibs.initialize
from nnlibs.commons.logs import pretty_json
from nnlibs.commons.library import (
    configure_directory,
    read_dataset,
    read_model,
)
from nnlibs.commons.maths import relu
from nnlibs.meta.models import EpyNN
from nnlibs.embedding.models import Embedding
from nnlibs.dense.models import Dense
from prepare_dataset import (
    labeled_dataset,
    unlabeled_dataset,
)
from settings import (
    dataset as se_dataset,
    config as se_config,
    hPars as se_hPars,
)


############################ HEADERS ##########################
random.seed(1)

np.set_printoptions(precision=3,threshold=sys.maxsize)

np.seterr(all='raise')

configure_directory(se_config=se_config)

settings = [se_dataset, se_config, se_hPars]

############################ DATASET ##########################
dataset = labeled_dataset(se_dataset)

embedding = Embedding(dataset, se_dataset)


############################ MODEL ############################
# Single-layer perceptron
dense = Dense()
name = 'Embedding_Dense'
model = EpyNN(layers=[embedding, dense], settings=settings, seed=1, name=name)

# Feed-forward Neural Network (Multi-layer perceptron)
#hidden_dense = Dense(nodes=4, activate=relu)
#name = 'Embedding_Dense-4-ReLU_Dense'
#model = EpyNN(layers=[embedding, hidden_dense, dense], settings=settings, seed=1, name=name)

# Feed-forward Neural Network (Deep Learning)
#more_hidden_dense = Dense(nodes=8, activate=relu)
#name = 'Embedding_Dense-8-ReLU_Dense-4-ReLU_Dense'
#model = EpyNN(layers=[embedding, more_hidden_dense, hidden_dense, dense], settings=settings, seed=1, name=name)

########################### TRAINING ###########################
model.train()
model.plot()
model.evaluate(write=True)


######################### PREDICTION ###########################
model = read_model()

unlabeled_more = unlabeled_dataset(N_SAMPLES=10)

dset = model.predict(unlabeled_more)

dset.P = np.argmax(dset.A, axis=1)

for i in range(len(unlabeled_more)):
    print(unlabeled_more[i], dset.A[i], dset.P[i])
