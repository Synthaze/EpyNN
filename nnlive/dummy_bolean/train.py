#EpyNN/nnlive/dummy_bolean/train.py
# Set environment and import default settings
from nnlibs.initialize import *
# Import common models
from nnlibs.commons.models import runData, hPars
# Import EpyNN meta-model to build neural networks
from nnlibs.meta.models import EpyNN
# Import models specific to layer architectures
from nnlibs.conv.models import Convolution
from nnlibs.dense.models import Dense
from nnlibs.dropout.models import Dropout
from nnlibs.flatten.models import Flatten
from nnlibs.gru.models import GRU
from nnlibs.lstm.models import LSTM
from nnlibs.pool.models import Pooling
from nnlibs.rnn.models import RNN
# Import utils
import nnlibs.commons.library as cl
# Import EpyNN math library
import nnlibs.commons.maths as cm
# Import data-specific routine to prepare sets
import sets_prepare as sp
# Import Numpy for seed initialization
import numpy as np
# Import random for seed initialization
import random
# Import local EpyNN settings
import settings


############################ INITIALIZATION ############################
# Facultative, to display numpy array entirely (truncated by default)
np.set_printoptions(precision=3,threshold=sys.maxsize)
#
np.seterr(all='raise')
# For reproducibility
np.random.seed(1)
random.seed(1)
## See EpyNN/nnlibs/initialize.py
# Build default EpyNN directory
cl.init_dir(settings.config)
## See EpyNN/nnlibs/commons/models.py
# Initialize runData model
runData = runData(settings.config)
# Initialize hPars (hyperparameters) model
hPars = hPars(settings.hPars)
########################################################################


############################### DATASETS ###############################
## See ./sets_prepare.py
# Retrieve datasets with dsets = [training,testing,validation]
dsets = sp.sets_prepare(runData)
########################################################################


########################## INITIALIZE_LAYERS ###########################
## Dense layer - see EpyNN/nnlibs/dense/models.py ######################
# Dense() layer takes X data of shape (a,d) - 2D
dense = Dense(16,cm.relu) # Custom (num_neurons=16,activate=cm.relu)
out_dense = Dense(2,cm.softmax) # Custom (num_neurons=10,activate=cm.softmax)
########################################################################


######################### BUILD_ARCHITECTURES ###########################
## Single_Layer_Perceptron ##############################################
name = 'Single_Layer_Perceptron'
layers = [out_dense]
## Feed_Forward_Neural_Network ##########################################
#name = 'Feed_Forward_Neural_Network'
#layers = [dense,out_dense]
#########################################################################


########################### NEURAL_NETWORK #############################
# Initialize with EpyNN meta-model - see EpyNN/nnlibs/meta/models.py
model = EpyNN(name=name,layers=layers)
# Train your model - see EpyNN/nnlibs/meta/train.py
model.train(dsets,hPars,runData)
# Plot results
model.plot(hPars,runData)
########################################################################
