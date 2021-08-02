#################################
from timeit import default_timer as timer
from termcolor import cprint

SEED = 1

N_EPOCHS = 3

LRATE = 1

batch_size = 60000
#################################
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

np.random.seed(SEED)
tf.random.set_seed(SEED)

np.seterr(all='warn')

# Disable GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Hide tensorflow debug info on call
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


tf.keras.backend.set_floatx('float64')

opt = tf.keras.optimizers.SGD(learning_rate=LRATE)
keras_loss_function = tf.keras.losses.CategoricalCrossentropy()

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float64") / 255
x_test = x_test.astype("float64") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


####################################
keras_model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
#        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="sigmoid"),
    ]
)

keras_model.summary()

x_train = x_train[:10]
y_train = y_train[:10]

keras_model.compile(loss=keras_loss_function, optimizer=opt, metrics=["accuracy"])

for layer in keras_model.layers:
    layer.trainable = False

keras_model.fit(x_train, y_train, batch_size=batch_size, epochs=N_EPOCHS)

weights = []

for layer in keras_model.layers:
    weights.append(layer.get_weights())

for layer in keras_model.layers:
    layer.trainable = True

keras_model.compile(loss=keras_loss_function, optimizer=opt, metrics=["accuracy"])

keras_model.fit(x_train, y_train, batch_size=batch_size, epochs=N_EPOCHS)

keras_preds = keras_model.predict(x_train)
####################################


from nnlibs.meta.models import EpyNN
from nnlibs.commons.models import dataSet
from nnlibs.embedding.models import Embedding
from nnlibs.convolution.models import Convolution
from nnlibs.pooling.models import Pooling
from nnlibs.flatten.models import Flatten
from nnlibs.dense.models import Dense
from settings import (
    dataset as se_dataset,
    config as se_config,
    hPars as se_hPars
)


settings = [se_dataset, se_config, se_hPars]

embedding = Embedding()

dset = dataSet([])

dset.X = x_train
dset.Y = y_train
dset.ids = [i for i in range (x_train.shape[0])]

embedding.single = True

embedding.dtrain = dset
embedding.batch_dtrain = [dset]
embedding.dsets = [dset]

se_config['training_epochs'] = N_EPOCHS
se_config['metrics_list'][1] = 'CCE'
se_config['training_loss'] = 'CCE'
se_hPars['learning_rate'] = LRATE


layers = [
    embedding,
    Convolution(16, 3),
    Pooling(14),
    Flatten(),
    Dense(num_classes)
]


epynn_model = EpyNN(layers=layers, settings=settings)

#epynn_model.train(run=False)

epynn_model.train(run=False)

epynn_model.layers[1].p['W'] = weights[0][0]
#epynn_model.layers[1].p['b'] = weights[2][1]


epynn_model.train(init=False)


epynn_preds = epynn_model.forward(embedding.dtrain.X)


############################ Compare ############################

cprint('\nKeras loss function (CCE) applied on Keras and EpyNN output probs:', attrs=['bold'])
print(keras_loss_function(y_train, keras_preds), '(Keras)')
print(keras_loss_function(y_train, epynn_preds), '(EpyNN)', end='\n\n')

cprint('\nEpyNN loss function (CCE) applied on Keras and EpyNN output probs:', attrs=['bold'])
print(epynn_model.training_loss(y_train, keras_preds).mean(axis=1), '(Keras)')
print(epynn_model.training_loss(y_train, epynn_preds).mean(axis=1), '(EpyNN)', end='\n\n')

cprint('Logits from output layer in Keras and EpyNN:', attrs=['bold'])
print(keras_preds, '(Keras)')
print(epynn_preds, '(EpyNN)', end='\n\n')

cprint('Accuracy from Keras and EpyNN:', attrs=['bold'])
print(np.mean(np.argmax(keras_preds, axis=1) == np.argmax(y_train, axis=1)), '(Keras)')
print(np.mean(np.argmax(epynn_preds, axis=1) == np.argmax(y_train, axis=1)), '(EpyNN)', end='\n\n')

# cprint('Total CPU time:', attrs=['bold'])
# print(keras_end, '(Keras)')
# print(epynn_end, '(EpyNN)', end='\n\n')
