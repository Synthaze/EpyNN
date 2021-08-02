"""
This will run layers = [input, RNN] with RNN(20) in many-to-one mode

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
N_SAMPLES = 5      # m
LEN_SEQ = 5       # s
V_SIZE = 20        # v Null

N_CELLS = 5

# Labels will be one-hot encoded
OUTPUT_NODES = N_LABELS = 2

N_EPOCHS = 10

LRATE = 0.1


######################## SHARED DATA ##########################
# Input data of shape (m, n)
X_train = np.random.standard_normal((N_SAMPLES, LEN_SEQ, V_SIZE))

# Average of features for each sample of shape (m, s, v) -> (m,)
ave_features = np.mean(X_train, axis=(2, 1))
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
from tensorflow.keras.layers import Flatten as KFlatten
from tensorflow.keras.layers import SimpleRNN as KRNN
from tensorflow.keras import Sequential

tf.keras.backend.set_floatx('float64')
tf.random.set_seed(SEED)

initializer = tf.keras.initializers.GlorotNormal(seed=SEED)
opt = tf.keras.optimizers.SGD(learning_rate=LRATE)
keras_loss_function = tf.keras.losses.BinaryCrossentropy(reduction='none')

# Layers
keras_RNN = KRNN(N_CELLS, kernel_initializer=initializer, return_sequences=True, batch_input_shape=(N_SAMPLES, LEN_SEQ, V_SIZE), activation='tanh')
keras_flatten = KFlatten()
keras_dense = KDense(OUTPUT_NODES, activation='sigmoid', kernel_initializer=initializer)

# Model
keras_model = Sequential()
keras_model.add(keras_RNN)
keras_model.add(keras_flatten)
keras_model.add(keras_dense)

####### LAYERS ARE FROZEN
keras_model.layers[0].trainable = False
keras_model.layers[1].trainable = False
keras_model.layers[2].trainable = False
#keras_model.layers[3].trainable = False
keras_model.compile(optimizer=opt, loss=keras_loss_function, metrics=['accuracy'])

keras_model.fit(X_train, y_train, epochs=1, batch_size=None, verbose=1)
keras_model.summary()

# Get Params
rnn_weights = keras_model.layers[0].get_weights()
shared_R_U = rnn_weights[0]
shared_R_W = rnn_weights[1]
shared_R_b = rnn_weights[2]

shared_W = keras_model.layers[2].get_weights()[0]
shared_b =  keras_model.layers[2].get_weights()[1]
# This is immutable whatever happens
checksum_W = np.sum(shared_W)
checksum_b = np.sum(shared_b)

print('shared_R_U.shape:', shared_R_U.shape)
print('shared_R_U[0]:', shared_R_U[0])
print('shared_R_W.shape:', shared_R_W.shape)
print('shared_R_W[0]:', shared_R_W[0])
print('shared_R_b.shape:',shared_R_b.shape)
print('shared_R_b[0]',shared_R_b[0], end='\n\n')

print('shared_W.shape:', shared_W.shape)
print('shared_W[0]:', shared_W[0])
print('shared_b.shape:',shared_b.shape)
print('shared_b[0]',shared_b[0], end='\n\n')


###############################################################
############################ TF/KERAS #########################

####### LAYERS ARE UNFROZEN
keras_model.layers[0].trainable = True
keras_model.layers[1].trainable = True
keras_model.layers[2].trainable = True
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
from nnlibs.flatten.models import Flatten as EFlatten
from nnlibs.rnn.models import RNN as ERNN
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

epynn_rnn = ERNN(N_CELLS)
epynn_rnn.p['U'] = shared_R_U
epynn_rnn.p['W'] = shared_R_W
epynn_rnn.p['b'] = shared_R_b


flatten = EFlatten()

epynn_dense = EDense(nodes=OUTPUT_NODES, activate=sigmoid)
epynn_dense.p['W'] = shared_W
epynn_dense.p['b'] = shared_b

epynn_model = EpyNN(layers=[embedding, epynn_rnn, flatten, epynn_dense], settings=settings, seed=SEED)

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

X_train.shape: (10, 20, 5)
X_train[0]: [[ 1.62434536 -0.61175641 -0.52817175 -1.07296862  0.86540763]
 [-2.3015387   1.74481176 -0.7612069   0.3190391  -0.24937038]
 [ 1.46210794 -2.06014071 -0.3224172  -0.38405435  1.13376944]
 [-1.09989127 -0.17242821 -0.87785842  0.04221375  0.58281521]
 [-1.10061918  1.14472371  0.90159072  0.50249434  0.90085595]
 [-0.68372786 -0.12289023 -0.93576943 -0.26788808  0.53035547]
 [-0.69166075 -0.39675353 -0.6871727  -0.84520564 -0.67124613]
 [-0.0126646  -1.11731035  0.2344157   1.65980218  0.74204416]
 [-0.19183555 -0.88762896 -0.74715829  1.6924546   0.05080775]
 [-0.63699565  0.19091548  2.10025514  0.12015895  0.61720311]
 [ 0.30017032 -0.35224985 -1.1425182  -0.34934272 -0.20889423]
 [ 0.58662319  0.83898341  0.93110208  0.28558733  0.88514116]
 [-0.75439794  1.25286816  0.51292982 -0.29809284  0.48851815]
 [-0.07557171  1.13162939  1.51981682  2.18557541 -1.39649634]
 [-1.44411381 -0.50446586  0.16003707  0.87616892  0.31563495]
 [-2.02220122 -0.30620401  0.82797464  0.23009474  0.76201118]
 [-0.22232814 -0.20075807  0.18656139  0.41005165  0.19829972]
 [ 0.11900865 -0.67066229  0.37756379  0.12182127  1.12948391]
 [ 1.19891788  0.18515642 -0.37528495 -0.63873041  0.42349435]
 [ 0.07734007 -0.34385368  0.04359686 -0.62000084  0.69803203]]
y_train.shape: (10, 2)
y_train[0] [1. 0.]

1/1 [==============================] - 0s 337us/step - loss: 1.2258 - accuracy: 0.4000
shared_W.shape: (100, 2)
shared_W[0]: [ 0.13752819 -0.0181232 ]
shared_W[1]: [-0.22954875  0.03672915]
shared_b.shape: (2,)
shared_b[0] 0.0

Epoch 1/10
1/1 [==============================] - 0s 355us/step - loss: 1.2258 - accuracy: 0.4000
Epoch 2/10
1/1 [==============================] - 0s 254us/step - loss: 1.0022 - accuracy: 0.4000
Epoch 3/10
1/1 [==============================] - 0s 238us/step - loss: 0.8189 - accuracy: 0.5000
Epoch 4/10
1/1 [==============================] - 0s 230us/step - loss: 0.6732 - accuracy: 0.7000
Epoch 5/10
1/1 [==============================] - 0s 226us/step - loss: 0.5593 - accuracy: 0.8000
Epoch 6/10
1/1 [==============================] - 0s 236us/step - loss: 0.4706 - accuracy: 0.8000
Epoch 7/10
1/1 [==============================] - 0s 232us/step - loss: 0.4012 - accuracy: 0.9000
Epoch 8/10
1/1 [==============================] - 0s 306us/step - loss: 0.3467 - accuracy: 1.0000
Epoch 9/10
1/1 [==============================] - 0s 284us/step - loss: 0.3035 - accuracy: 1.0000
Epoch 10/10
1/1 [==============================] - 0s 274us/step - loss: 0.2689 - accuracy: 1.0000

checksum_W versus epynn_dense_W in EpyNN model: True
checksum_b versus epynn_dense_b in EpyNN model: True

+-------+----------+----------+-------+------------------+
| epoch |  lrate   | accuracy |  BCE  |    Experiment    |
|       |  Dense   |   (1)    |  (1)  |                  |
+-------+----------+----------+-------+------------------+
|   0   | 1.00e-02 |  0.400   | 1.002 | 1627846944_Model |
|   1   | 1.00e-02 |  0.500   | 0.819 | 1627846944_Model |
|   2   | 1.00e-02 |  0.700   | 0.673 | 1627846944_Model |
|   3   | 1.00e-02 |  0.800   | 0.559 | 1627846944_Model |
|   4   | 1.00e-02 |  0.800   | 0.471 | 1627846944_Model |
|   5   | 1.00e-02 |  0.900   | 0.401 | 1627846944_Model |
|   6   | 1.00e-02 |  1.000   | 0.347 | 1627846944_Model |
|   7   | 1.00e-02 |  1.000   | 0.303 | 1627846944_Model |
|   8   | 1.00e-02 |  1.000   | 0.269 | 1627846944_Model |
|   9   | 1.00e-02 |  1.000   | 0.241 | 1627846944_Model |
+-------+----------+----------+-------+------------------+
TIME: 0.46s RATE: 21.74e/s TTC: 0s

Keras loss function (BCE) applied on Keras and EpyNN output probs:
tf.Tensor(
[0.30036414 0.46518841 0.23198201 0.15970144 0.06697492 0.33348233
 0.23714679 0.12541832 0.26913121 0.21977907], shape=(10,), dtype=float64) (Keras)
tf.Tensor(
[0.30036414 0.46518838 0.23198201 0.15970144 0.06697492 0.33348233
 0.23714678 0.12541831 0.26913121 0.21977907], shape=(10,), dtype=float64) (EpyNN)


EpyNN loss function (BCE) applied on Keras and EpyNN output probs:
[0.30036427 0.46518857 0.23198213 0.15970156 0.06697503 0.33348246
 0.23714692 0.12541843 0.26913135 0.2197792 ] (Keras)
[0.30036426 0.46518855 0.23198213 0.15970156 0.06697503 0.33348245
 0.23714691 0.12541843 0.26913135 0.2197792 ] (EpyNN)

Logits from output layer in Keras and EpyNN:
[[0.73028522 0.24904416]
 [0.4954647  0.20396966]
 [0.20562339 0.79154649]
 [0.14982567 0.85462777]
 [0.91935141 0.04864046]
 [0.80690969 0.36391393]
 [0.20804615 0.78580887]
 [0.12994789 0.89437093]
 [0.79287627 0.26374195]
 [0.86340519 0.25374448]] (Keras)
[[0.73028523 0.24904416]
 [0.49546472 0.20396965]
 [0.20562338 0.7915465 ]
 [0.14982566 0.85462777]
 [0.91935141 0.04864046]
 [0.80690969 0.36391392]
 [0.20804614 0.78580888]
 [0.12994789 0.89437093]
 [0.79287628 0.26374194]
 [0.86340519 0.25374448]] (EpyNN)

Accuracy from Keras and EpyNN:
1.0 (Keras)
1.0 (EpyNN)

Total CPU time:
0.25941607799904887 (Keras)
0.005428262003988493 (EpyNN)

"""
