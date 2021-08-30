# EpyNN/nnlive/dummy_boolean/train.py
# Standard library imports
import random

# Related third party imports
import numpy as np

# Local application/library specific imports
import nnlibs.initialize
from nnlibs.commons.library import (
    configure_directory,
    read_model,
)
from nnlibs.network.models import EpyNN
from nnlibs.embedding.models import Embedding
from nnlibs.dense.models import Dense
from prepare_dataset import prepare_dataset


########################## CONFIGURE ##########################
random.seed(1)

np.set_printoptions(threshold=10)

np.seterr(all='warn')

configure_directory(clear=True)    # This is a dummy example


############################ DATASET ##########################
X_features, Y_label = prepare_dataset(N_SAMPLES=50)


####################### BUILD AND TRAIN MODEL #################

embedding = Embedding(X_data=X_features,
                      Y_data=Y_label,
                      relative_size=(2, 1, 0))


### Feed-Forward

# Model
name = 'Perceptron_Dense-1-sigmoid'

dense = Dense()

model = EpyNN(layers=[embedding, dense], name=name)

model.train(epochs=100)

model.plot(path=False)


########################## PREDICTION #########################
model.write()
# model.write(path=/your/custom/path)

model = read_model()
# model = read_model(path=/your/custom/path)

X_features, _ = prepare_dataset(N_SAMPLES=10)

dset = model.predict(X_features)

for n, pred, probs, features in zip(dset.ids, dset.P, dset.A, dset.X):
    print(n, pred, probs, features)
