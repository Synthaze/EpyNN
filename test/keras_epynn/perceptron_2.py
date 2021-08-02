"""
This will run layers = [input, output] with output = Dense(2)

Labels are one-hot encoded.

Config is identical between TF/Keras and EpyNN.

Expected results at the end of file.
"""
###############################################################
################ HEADERS and SHARED CST/PARS ##################
import os
import numpy as np
from timeit import default_timer as timer
from termcolor import cprint


# Disable GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Hide tensorflow debug info on call
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Seed everywhere
SEED = 1
np.random.seed(SEED)

#
np.seterr(all='warn')

# Dataset and samples
N_SAMPLES = 10      # m
N_FEATURES = 20     # n

# Labels will be one-hot encoded
OUTPUT_NODES = N_LABELS = 2

N_EPOCHS = 10

LRATE = 0.05


######################## SHARED DATA ##########################
# Input data of shape (m, n)
X_train = np.random.standard_normal((N_SAMPLES, N_FEATURES))

# Average of features for each sample of shape (m, n) -> (m,)
ave_features = np.mean(X_train, axis=1)
# Mean of averaged features of shape (m,) -> (1,)
mean_samples = np.mean(ave_features)

# Assign labels (m,)
y_train = np.where(ave_features < mean_samples, 0, 1)
# One-hot encode labels (m,) -> (m, 2)
y_train = np.array([[x, np.abs(1 - x)] for x in y_train], dtype=float)

print('\nX_train.shape:', X_train.shape)
print('X_train[0]:', X_train[0])
print('y_train.shape:',y_train.shape)
print('y_train[0]',y_train[0], end='\n\n')


#################### SHARED WEIGTHS/BIAS #######################
import tensorflow as tf
from tensorflow.keras.layers import Dense as KDense


tf.keras.backend.set_floatx('float64')

tf.random.set_seed(SEED)

# Xavier initializer
initializer = tf.keras.initializers.GlorotNormal(seed=SEED)

_ = KDense(OUTPUT_NODES, activation=None, kernel_initializer=initializer)
_(X_train)

shared_W = _.get_weights()[0]
shared_b = _.get_weights()[1]

# This is immutable whatever happens
checksum_W = np.sum(shared_W)
checksum_b = np.sum(shared_b)

del _

print('shared_W.shape:', shared_W.shape)
print('shared_W[0]:', shared_W[0])
print('shared_W[1]:', shared_W[1])
print('shared_b.shape:',shared_b.shape)
print('shared_b[0]',shared_b[0], end='\n\n')


###############################################################
############################ TF/KERAS #########################
from tensorflow.keras import Sequential


keras_loss_function = tf.keras.losses.BinaryCrossentropy(reduction='none')

opt = tf.keras.optimizers.SGD(learning_rate=LRATE)

keras_dense = KDense(OUTPUT_NODES, activation='sigmoid')
keras_dense(X_train)

keras_dense.set_weights([shared_W, shared_b])

keras_model = Sequential()
keras_model.add(keras_dense)

keras_model.compile(optimizer=opt, loss=keras_loss_function, metrics=['accuracy'])

cprint('\nchecksum_W versus keras_dense_W: %s' % (checksum_W == np.sum(keras_dense.get_weights()[0])), 'green', attrs=['bold'])
cprint('checksum_b versus keras_dense_b: %s' % (checksum_b == np.sum(keras_dense.get_weights()[1])), 'green', attrs=['bold'], end='\n\n')

# Training ###
keras_start = timer()
keras_model.fit(X_train, y_train, epochs=N_EPOCHS, batch_size=None, verbose=1)
keras_end = timer() - keras_start
##############

keras_preds = keras_model.predict(X_train)


############################ EpyNN ############################
from nnlibs.commons.maths import sigmoid
from nnlibs.commons.models import dataSet
from nnlibs.dense.models import Dense as EDense
from nnlibs.embedding.models import Embedding
from nnlibs.meta.models import EpyNN
from settings import (
    dataset as se_dataset,
    config as se_config,
    hPars as se_hPars
)


settings = [se_dataset, se_config, se_hPars]
se_config['training_epochs'] = N_EPOCHS
se_hPars['learning_rate'] = LRATE

dataset = [[x, y] for x,y in zip(X_train, y_train)]
embedding = Embedding(dataset, se_dataset, single=True)

epynn_dense = EDense(nodes=OUTPUT_NODES, activate=sigmoid)
epynn_dense.p['W'] = shared_W
epynn_dense.p['b'] = shared_b

epynn_model = EpyNN(layers=[embedding, epynn_dense], settings=settings, seed=SEED)

epynn_model.initialize(init_params=False, verbose=False)

cprint('\nchecksum_W versus epynn_dense_W in EpyNN model: %s' % (checksum_W == np.sum(epynn_dense.p['W'])), 'green', attrs=['bold'])
cprint('checksum_b versus epynn_dense_b in EpyNN model: %s' % (checksum_b == np.sum(epynn_dense.p['b'])), 'green', attrs=['bold'], end='\n\n')

# Training ###
epynn_start = timer()
epynn_model.train(init=False)
epynn_end = timer() - epynn_start
##############

epynn_preds = epynn_model.forward(embedding.dtrain.X)


############################ Compare ############################

cprint('\nKeras loss function (BCE) applied on Keras and EpyNN output probs:', attrs=['bold'])
print(keras_loss_function(y_train, keras_preds), '(Keras)')
print(keras_loss_function(y_train, epynn_preds), '(EpyNN)', end='\n\n')

cprint('\nEpyNN loss function (BCE) applied on Keras and EpyNN output probs:', attrs=['bold'])
print(epynn_model.training_loss(y_train, keras_preds).mean(axis=1), '(Keras)')
print(epynn_model.training_loss(y_train, epynn_preds).mean(axis=1), '(EpyNN)', end='\n\n')

cprint('Logits from output layer in Keras and EpyNN:', attrs=['bold'])
print(keras_preds, '(Keras)')
print(epynn_preds, '(EpyNN)', end='\n\n')

cprint('Accuracy from Keras and EpyNN:', attrs=['bold'])
print(np.mean(np.argmax(keras_preds, axis=1) == np.argmax(y_train, axis=1)), '(Keras)')
print(np.mean(np.argmax(epynn_preds, axis=1) == np.argmax(y_train, axis=1)), '(EpyNN)', end='\n\n')

cprint('Total CPU time:', attrs=['bold'])
print(keras_end, '(Keras)')
print(epynn_end, '(EpyNN)', end='\n\n')


"""

X_train.shape: (10, 20)
X_train[0]: [ 1.62434536 -0.61175641 -0.52817175 -1.07296862  0.86540763 -2.3015387
  1.74481176 -0.7612069   0.3190391  -0.24937038  1.46210794 -2.06014071
 -0.3224172  -0.38405435  1.13376944 -1.09989127 -0.17242821 -0.87785842
  0.04221375  0.58281521]
y_train.shape: (10, 2)
y_train[0] [0. 1.]

shared_W.shape: (20, 2)
shared_W[0]: [ 0.29612869 -0.03902327]
shared_W[1]: [-0.49426934  0.07908601]
shared_b.shape: (2,)
shared_b[0] 0.0


checksum_W versus keras_dense_W: True
checksum_b versus keras_dense_b: True

Epoch 1/10
1/1 [==============================] - 0s 339us/step - loss: 0.9160 - accuracy: 0.6000
Epoch 2/10
1/1 [==============================] - 0s 252us/step - loss: 0.6986 - accuracy: 0.7000
Epoch 3/10
1/1 [==============================] - 0s 228us/step - loss: 0.5460 - accuracy: 0.8000
Epoch 4/10
1/1 [==============================] - 0s 276us/step - loss: 0.4417 - accuracy: 0.9000
Epoch 5/10
1/1 [==============================] - 0s 248us/step - loss: 0.3701 - accuracy: 1.0000
Epoch 6/10
1/1 [==============================] - 0s 227us/step - loss: 0.3194 - accuracy: 1.0000
Epoch 7/10
1/1 [==============================] - 0s 224us/step - loss: 0.2821 - accuracy: 1.0000
Epoch 8/10
1/1 [==============================] - 0s 278us/step - loss: 0.2534 - accuracy: 1.0000
Epoch 9/10
1/1 [==============================] - 0s 237us/step - loss: 0.2306 - accuracy: 1.0000
Epoch 10/10
1/1 [==============================] - 0s 243us/step - loss: 0.2119 - accuracy: 1.0000

checksum_W versus epynn_dense_W in EpyNN model: True
checksum_b versus epynn_dense_b in EpyNN model: True

+-------+----------+----------+-------+------------------+
| epoch |  lrate   | accuracy |  BCE  |    Experiment    |
|       |  Dense   |   (1)    |  (1)  |                  |
+-------+----------+----------+-------+------------------+
|   0   | 5.00e-02 |  0.700   | 0.699 | 1627846993_Model |
|   1   | 5.00e-02 |  0.800   | 0.546 | 1627846993_Model |
|   2   | 5.00e-02 |  0.900   | 0.442 | 1627846993_Model |
|   3   | 5.00e-02 |  1.000   | 0.370 | 1627846993_Model |
|   4   | 5.00e-02 |  1.000   | 0.319 | 1627846993_Model |
|   5   | 5.00e-02 |  1.000   | 0.282 | 1627846993_Model |
|   6   | 5.00e-02 |  1.000   | 0.253 | 1627846993_Model |
|   7   | 5.00e-02 |  1.000   | 0.231 | 1627846993_Model |
|   8   | 5.00e-02 |  1.000   | 0.212 | 1627846993_Model |
|   9   | 5.00e-02 |  1.000   | 0.196 | 1627846993_Model |
+-------+----------+----------+-------+------------------+
TIME: 0.47s RATE: 21.28e/s TTC: 0s

Keras loss function (BCE) applied on Keras and EpyNN output probs:
tf.Tensor(
[0.12490708 0.16035704 0.1835856  0.12837207 0.40478012 0.37522024
 0.1429137  0.08213077 0.08784835 0.2736344 ], shape=(10,), dtype=float64) (Keras)
tf.Tensor(
[0.12490708 0.16035704 0.1835856  0.12837207 0.40478012 0.37522024
 0.1429137  0.08213077 0.08784836 0.2736344 ], shape=(10,), dtype=float64) (EpyNN)


EpyNN loss function (BCE) applied on Keras and EpyNN output probs:
[0.12490719 0.16035716 0.18358571 0.12837219 0.40478026 0.37522038
 0.14291382 0.08213089 0.08784846 0.27363454] (Keras)
[0.12490719 0.16035717 0.18358571 0.12837219 0.40478026 0.37522038
 0.14291382 0.08213089 0.08784846 0.27363455] (EpyNN)

Logits from output layer in Keras and EpyNN:
[[0.185802   0.95670262]
 [0.10111547 0.80725665]
 [0.81075059 0.14561777]
 [0.85707948 0.09743968]
 [0.39437969 0.73487234]
 [0.31583473 0.69012331]
 [0.85665762 0.12287932]
 [0.94216877 0.09939715]
 [0.93355019 0.10141715]
 [0.71070777 0.18598391]] (Keras)
[[0.18580201 0.95670262]
 [0.10111547 0.80725665]
 [0.81075058 0.14561778]
 [0.85707947 0.09743968]
 [0.39437969 0.73487234]
 [0.31583473 0.69012331]
 [0.85665762 0.12287932]
 [0.94216877 0.09939715]
 [0.93355019 0.10141716]
 [0.71070777 0.18598391]] (EpyNN)

Accuracy from Keras and EpyNN:
1.0 (Keras)
1.0 (EpyNN)

Total CPU time:
0.30516785199870355 (Keras)
0.005076708999695256 (EpyNN)


"""
