"""
This will run layers = [input, output] with output = Dense(1)

Labels are not one-hot encoded.

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
N_SAMPLES = 30      # m
N_FEATURES = 20     # n

OUTPUT_NODES = 1

N_EPOCHS = 10

LRATE = 0.1


######################## SHARED DATA ##########################
# Input data of shape (m, n)
X_train = np.random.standard_normal((N_SAMPLES, N_FEATURES))

# Average of features for each sample of shape (m, n) -> (m,)
ave_features = np.mean(X_train, axis=1)
# Mean of averaged features of shape (m,) -> (1,)
mean_samples = np.mean(ave_features)

# Assign labels (m,)
y_train = np.where(ave_features < mean_samples, 0, 1)

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

y_train = y_train.reshape(N_SAMPLES, 1)
cprint('Accuracy from Keras and EpyNN:', attrs=['bold'])
print(np.mean(np.around(keras_preds) == y_train), '(Keras)')
print(np.mean(np.around(epynn_preds) == y_train), '(EpyNN)', end='\n\n')

cprint('Total CPU time:', attrs=['bold'])
print(keras_end, '(Keras)')
print(epynn_end, '(EpyNN)', end='\n\n')


"""

X_train.shape: (30, 20)
X_train[0]: [ 1.62434536 -0.61175641 -0.52817175 -1.07296862  0.86540763 -2.3015387
  1.74481176 -0.7612069   0.3190391  -0.24937038  1.46210794 -2.06014071
 -0.3224172  -0.38405435  1.13376944 -1.09989127 -0.17242821 -0.87785842
  0.04221375  0.58281521]
y_train.shape: (30,)
y_train[0] 0

shared_W.shape: (20, 1)
shared_W[0]: [0.30309738]
shared_W[1]: [-0.0399416]
shared_b.shape: (1,)
shared_b[0] 0.0


checksum_W versus keras_dense_W: True
checksum_b versus keras_dense_b: True

Epoch 1/10
1/1 [==============================] - 0s 379us/step - loss: 0.8641 - accuracy: 0.4667
Epoch 2/10
1/1 [==============================] - 0s 241us/step - loss: 0.4336 - accuracy: 0.8000
Epoch 3/10
1/1 [==============================] - 0s 250us/step - loss: 0.2745 - accuracy: 0.9667
Epoch 4/10
1/1 [==============================] - 0s 238us/step - loss: 0.2226 - accuracy: 0.9333
Epoch 5/10
1/1 [==============================] - 0s 253us/step - loss: 0.1917 - accuracy: 0.9667
Epoch 6/10
1/1 [==============================] - 0s 236us/step - loss: 0.1700 - accuracy: 0.9667
Epoch 7/10
1/1 [==============================] - 0s 238us/step - loss: 0.1537 - accuracy: 0.9667
Epoch 8/10
1/1 [==============================] - 0s 324us/step - loss: 0.1407 - accuracy: 1.0000
Epoch 9/10
1/1 [==============================] - 0s 288us/step - loss: 0.1300 - accuracy: 1.0000
Epoch 10/10
1/1 [==============================] - 0s 253us/step - loss: 0.1210 - accuracy: 1.0000

checksum_W versus epynn_dense_W in EpyNN model: True
checksum_b versus epynn_dense_b in EpyNN model: True

+-------+----------+----------+-------+------------------+
| epoch |  lrate   | accuracy |  BCE  |    Experiment    |
|       |  Dense   |   (1)    |  (1)  |                  |
+-------+----------+----------+-------+------------------+
|   0   | 1.00e-01 |  0.800   | 0.434 | 1627846972_Model |
|   1   | 1.00e-01 |  0.967   | 0.275 | 1627846972_Model |
|   2   | 1.00e-01 |  0.933   | 0.223 | 1627846972_Model |
|   3   | 1.00e-01 |  0.967   | 0.192 | 1627846972_Model |
|   4   | 1.00e-01 |  0.967   | 0.170 | 1627846972_Model |
|   5   | 1.00e-01 |  0.967   | 0.154 | 1627846972_Model |
|   6   | 1.00e-01 |  1.000   | 0.141 | 1627846972_Model |
|   7   | 1.00e-01 |  1.000   | 0.130 | 1627846972_Model |
|   8   | 1.00e-01 |  1.000   | 0.121 | 1627846972_Model |
|   9   | 1.00e-01 |  1.000   | 0.113 | 1627846972_Model |
+-------+----------+----------+-------+------------------+
TIME: 0.55s RATE: 18.18e/s TTC: 0s

Keras loss function (BCE) applied on Keras and EpyNN output probs:
tf.Tensor(
[1.19634163 1.11252916 1.07828498 1.08637106 0.70603848 1.57652152
 1.06536019 1.70239913 1.13779759 1.82483816 1.73926425 1.78461063
 0.93067938 1.67204714 0.69876587 2.19274473 3.13500524 1.58093894
 1.12827981 1.56149769 2.58090734 2.62589502 0.98005885 0.75201583
 1.22358716 1.18161571 1.84197295 1.43317139 4.2434659  1.59508407], shape=(30,), dtype=float64) (Keras)
tf.Tensor(
[1.19634163 1.11252916 1.07828498 1.08637106 0.70603848 1.57652152
 1.06536019 1.70239913 1.13779759 1.82483816 1.73926425 1.78461051
 0.93067938 1.67204714 0.69876587 2.19274473 3.13500524 1.58093894
 1.12827981 1.56149769 2.58090711 2.62589502 0.98005885 0.75201583
 1.22358716 1.18161571 1.84197295 1.43317127 4.2434659  1.59508407], shape=(30,), dtype=float64) (EpyNN)


EpyNN loss function (BCE) applied on Keras and EpyNN output probs:
[1.19634228 1.11252964 1.07828544 1.0863715  0.70603867 1.57652285
 1.06536061 1.70240045 1.13779806 1.82483984 1.73926621 1.78461209
 0.9306797  1.67204832 0.69876608 2.192748   3.1350439  1.5809403
 1.12828037 1.56149903 2.58091899 2.6259024  0.98005921 0.75201608
 1.22358776 1.18161635 1.8419746  1.43317234 4.24361786 1.59508514] (Keras)
[1.19634227 1.11252964 1.07828543 1.08637149 0.70603867 1.57652285
 1.0653606  1.70240044 1.13779805 1.82483983 1.7392662  1.78461208
 0.9306797  1.67204832 0.69876607 2.19274799 3.13504388 1.58094029
 1.12828037 1.56149902 2.58091897 2.62590238 0.98005921 0.75201608
 1.22358776 1.18161635 1.84197459 1.43317233 4.24361783 1.59508514] (EpyNN)

Logits from output layer in Keras and EpyNN:
[[0.08528915]
 [0.10458647]
 [0.84684909]
 [0.84957753]
 [0.55342376]
 [0.03554704]
 [0.84236103]
 [0.95731159]
 [0.86562266]
 [0.9663459 ]
 [0.02476459]
 [0.96361946]
 [0.16792908]
 [0.95470373]
 [0.40477569]
 [0.98337307]
 [0.00121069]
 [0.03519756]
 [0.10060387]
 [0.03676302]
 [0.00398196]
 [0.99267995]
 [0.14679977]
 [0.30008133]
 [0.0799155 ]
 [0.08836265]
 [0.96744226]
 [0.04911834]
 [0.99964961]
 [0.94731511]] (Keras)
[[0.08528915]
 [0.10458647]
 [0.84684909]
 [0.84957753]
 [0.55342375]
 [0.03554704]
 [0.84236103]
 [0.95731159]
 [0.86562265]
 [0.9663459 ]
 [0.0247646 ]
 [0.96361946]
 [0.16792908]
 [0.95470373]
 [0.40477569]
 [0.98337307]
 [0.00121069]
 [0.03519756]
 [0.10060387]
 [0.03676303]
 [0.00398196]
 [0.99267995]
 [0.14679978]
 [0.30008133]
 [0.0799155 ]
 [0.08836265]
 [0.96744226]
 [0.04911834]
 [0.99964961]
 [0.94731511]] (EpyNN)

Accuracy from Keras and EpyNN:
1.0 (Keras)
1.0 (EpyNN)

Total CPU time:
0.3305138440045994 (Keras)
0.005444635004096199 (EpyNN)


"""
