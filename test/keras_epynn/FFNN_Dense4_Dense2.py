"""
This will run layers = [input, hidden, output] with hidden = Dense(4) and output = Dense(2)

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

HIDDEN_NODES = 4

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
"""
We will prepare layers and freeze them, build keras model with lrate = 0 (in case), apply a dummy fit, get the parameters out and proceed.
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense as KDense
from tensorflow.keras import Sequential

tf.keras.backend.set_floatx('float64')
tf.random.set_seed(SEED)

initializer = tf.keras.initializers.GlorotNormal(seed=SEED)
opt = tf.keras.optimizers.SGD(learning_rate=LRATE)
keras_loss_function = tf.keras.losses.BinaryCrossentropy(reduction='none')

# Layers
keras_h_dense = KDense(HIDDEN_NODES, activation='relu', kernel_initializer=initializer)
keras_h_dense(X_train)
keras_dense = KDense(OUTPUT_NODES, activation='sigmoid', kernel_initializer=initializer)

# Model
keras_model = Sequential()
keras_model.add(keras_h_dense)
keras_model.add(keras_dense)

####### LAYERS ARE FROZEN
keras_model.layers[0].trainable = False
keras_model.layers[1].trainable = False
keras_model.compile(optimizer=opt, loss=keras_loss_function, metrics=['accuracy'])
keras_model.fit(X_train, y_train, epochs=1, batch_size=None, verbose=1)

####### Weight/Bias have been initialized but NOT trained
shared_h_W = keras_model.layers[0].get_weights()[0]
shared_h_b =  keras_model.layers[0].get_weights()[1]
# This is immutable whatever happens
checksum_h_W = np.sum(shared_h_W)
checksum_h_b = np.sum(shared_h_b)

shared_W = keras_model.layers[1].get_weights()[0]
shared_b =  keras_model.layers[1].get_weights()[1]
# This is immutable whatever happens
checksum_W = np.sum(shared_W)
checksum_b = np.sum(shared_b)

print('shared_h_W.shape:', shared_h_W.shape)
print('shared_h_W[0]:', shared_h_W[0])
print('shared_h_W[1]:', shared_h_W[1])
print('shared_h_b.shape:',shared_h_b.shape)
print('shared_h_b[0]',shared_h_b[0], end='\n\n')

print('shared_W.shape:', shared_W.shape)
print('shared_W[0]:', shared_W[0])
print('shared_W[1]:', shared_W[1])
print('shared_b.shape:',shared_b.shape)
print('shared_b[0]',shared_b[0], end='\n\n')


###############################################################
############################ TF/KERAS #########################

####### LAYERS ARE UNFROZEN
keras_model.layers[0].trainable = True
keras_model.layers[1].trainable = True
keras_model.compile(optimizer=opt, loss=keras_loss_function, metrics=['accuracy'])

# Training ###
keras_start = timer()
keras_model.fit(X_train, y_train, epochs=N_EPOCHS, batch_size=None, verbose=1)
keras_end = timer() - keras_start
##############

keras_preds = keras_model.predict(X_train)


############################ EpyNN ############################
from nnlibs.commons.maths import sigmoid, relu
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

epynn_h_dense = EDense(nodes=HIDDEN_NODES, activate=relu)
epynn_h_dense.p['W'] = shared_h_W
epynn_h_dense.p['b'] = shared_h_b

epynn_dense = EDense(nodes=OUTPUT_NODES, activate=sigmoid)
epynn_dense.p['W'] = shared_W
epynn_dense.p['b'] = shared_b

epynn_model = EpyNN(layers=[embedding, epynn_h_dense, epynn_dense], settings=settings, seed=SEED)

epynn_model.initialize(init_params=False, verbose=False)

cprint('\nchecksum_h_W versus epynn_dense_h_W in EpyNN model: %s' % (checksum_h_W == np.sum(epynn_h_dense.p['W'])), 'green', attrs=['bold'])
cprint('checksum_h_b versus epynn_dense_h_b in EpyNN model: %s' % (checksum_h_b == np.sum(epynn_h_dense.p['b'])), 'green', attrs=['bold'], end='\n')
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
print(epynn_model.training_loss(y_train, epynn_preds).mean(axis=1).mean(), '(EpyNN)', end='\n\n')

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

1/1 [==============================] - 0s 340us/step - loss: 0.8567 - accuracy: 0.3000
shared_h_W.shape: (20, 4)
shared_h_W[0]: [ 0.28352164 -0.03736194 -0.47322687  0.07571909]
shared_h_W[1]: [ 0.14666732  0.07194907 -0.29138584 -0.43638748]
shared_h_b.shape: (4,)
shared_h_b[0] 0.0

shared_W.shape: (4, 2)
shared_W[0]: [ 0.56704327 -0.07472388]
shared_W[1]: [-0.94645373  0.15143818]
shared_b.shape: (2,)
shared_b[0] 0.0

Epoch 1/10
1/1 [==============================] - 0s 397us/step - loss: 0.8567 - accuracy: 0.3000
Epoch 2/10
1/1 [==============================] - 0s 248us/step - loss: 0.7375 - accuracy: 0.5000
Epoch 3/10
1/1 [==============================] - 0s 259us/step - loss: 0.6810 - accuracy: 0.6000
Epoch 4/10
1/1 [==============================] - 0s 232us/step - loss: 0.6425 - accuracy: 0.8000
Epoch 5/10
1/1 [==============================] - 0s 263us/step - loss: 0.6116 - accuracy: 0.9000
Epoch 6/10
1/1 [==============================] - 0s 255us/step - loss: 0.5835 - accuracy: 0.9000
Epoch 7/10
1/1 [==============================] - 0s 261us/step - loss: 0.5604 - accuracy: 0.9000
Epoch 8/10
1/1 [==============================] - 0s 288us/step - loss: 0.5372 - accuracy: 0.9000
Epoch 9/10
1/1 [==============================] - 0s 277us/step - loss: 0.5122 - accuracy: 0.9000
Epoch 10/10
1/1 [==============================] - 0s 252us/step - loss: 0.4912 - accuracy: 0.9000

checksum_h_W versus epynn_dense_h_W in EpyNN model: True
checksum_h_b versus epynn_dense_h_b in EpyNN model: True

checksum_W versus epynn_dense_W in EpyNN model: True
checksum_b versus epynn_dense_b in EpyNN model: True

+-------+----------+----------+----------+-------+------------------+
| epoch |  lrate   |  lrate   | accuracy |  BCE  |    Experiment    |
|       |  Dense   |  Dense   |   (1)    |  (1)  |                  |
+-------+----------+----------+----------+-------+------------------+
|   0   | 5.00e-02 | 5.00e-02 |  0.500   | 0.738 | 1627846911_Model |
|   1   | 5.00e-02 | 5.00e-02 |  0.600   | 0.681 | 1627846911_Model |
|   2   | 5.00e-02 | 5.00e-02 |  0.800   | 0.643 | 1627846911_Model |
|   3   | 5.00e-02 | 5.00e-02 |  0.900   | 0.612 | 1627846911_Model |
|   4   | 5.00e-02 | 5.00e-02 |  0.900   | 0.583 | 1627846911_Model |
|   5   | 5.00e-02 | 5.00e-02 |  0.900   | 0.560 | 1627846911_Model |
|   6   | 5.00e-02 | 5.00e-02 |  0.900   | 0.537 | 1627846911_Model |
|   7   | 5.00e-02 | 5.00e-02 |  0.900   | 0.512 | 1627846911_Model |
|   8   | 5.00e-02 | 5.00e-02 |  0.900   | 0.491 | 1627846911_Model |
|   9   | 5.00e-02 | 5.00e-02 |  0.900   | 0.472 | 1627846911_Model |
+-------+----------+----------+----------+-------+------------------+
TIME: 0.42s RATE: 23.81e/s TTC: 0s

Keras loss function (BCE) applied on Keras and EpyNN output probs:
tf.Tensor(
[0.20786749 0.22001755 0.42833421 0.58701903 0.93890411 0.34877703
 0.57469821 0.58701903 0.37147012 0.45911935], shape=(10,), dtype=float64) (Keras)
tf.Tensor(
[0.20786749 0.22001755 0.42833421 0.58701909 0.93890411 0.34877706
 0.57469821 0.58701909 0.37147012 0.45911935], shape=(10,), dtype=float64) (EpyNN)


EpyNN loss function (BCE) applied on Keras and EpyNN output probs:
[0.20786761 0.22001768 0.42833435 0.58701924 0.93890437 0.34877718
 0.5746984  0.58701924 0.37147027 0.4591195 ] (Keras)
0.47232278616756906 (EpyNN)

Logits from output layer in Keras and EpyNN:
[[0.11743395 0.74765504]
 [0.11032777 0.72387743]
 [0.5838389  0.27278893]
 [0.5712836  0.45890962]
 [0.58568664 0.36910428]
 [0.22724288 0.64418851]
 [0.53590765 0.40880145]
 [0.5712836  0.45890962]
 [0.84062305 0.43409474]
 [0.76989235 0.48145809]] (Keras)
[[0.11743395 0.74765504]
 [0.11032777 0.72387742]
 [0.5838389  0.27278893]
 [0.5712836  0.45890962]
 [0.58568664 0.36910429]
 [0.22724288 0.64418851]
 [0.53590765 0.40880145]
 [0.5712836  0.45890962]
 [0.84062305 0.43409475]
 [0.76989235 0.48145809]] (EpyNN)

Accuracy from Keras and EpyNN:
0.9 (Keras)
0.9 (EpyNN)

Total CPU time:
0.27079277399752755 (Keras)
0.005861378995177802 (EpyNN)



"""
