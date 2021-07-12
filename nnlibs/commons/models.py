#EpyNN/nnlibs/commons/models.py
from nnlibs.commons.decorators import *
import nnlibs.commons.schedule as cs
import nnlibs.commons.maths as cm

import numpy as np
import time


class runData:

    @log_method
    def __init__(self,se_datasets,se_config):

        ## Metadata
        self.m = {}
        # Experiment Name
        self.m['n'] = se_config['experiment_name']
        # Unique identifier
        self.m['t'] = str(int(time.time()))
        # Experiment identifier
        self.m['nt'] = self.m['n']+'_'+self.m['t']
        # Logs frequency
        self.m['f'] = se_config['logs_frequency']
        # Logs frequency
        self.m['fd'] = se_config['logs_frequency_display']
        # Target dataset
        self.m['d'] = se_config['dataset_target']
        # Target metrics
        self.m['m'] = se_config['metrics_target']
        # Metrics to plot
        self.m['p'] = se_config['metrics_plot']
        # Number of samples
        self.m['s'] = se_datasets['N_SAMPLES']

        ## Bolean for display and write on disk actions
        self.b = {}
        # Save model parameters and associated data
        self.b['ms'] = se_config['model_save']
        # Display plot at the end of training
        self.b['pd'] = se_config['plot_display']
        # Save plot at the end of training
        self.b['ps'] = se_config['plot_save']
        # Flag on save
        self.b['s'] = False
        # Flag init run
        self.b['i'] = True

        ## Paths
        self.p = {}
        # Save model parameters and associated data
        self.p['ms'] = './models/'+self.m['n']+'_'+self.m['t']+'.pickle'
        # Save plots
        self.p['ps'] = './models/'+self.m['n']+'_'+self.m['t']+'.png'

        ## Metrics for each dataset
        self.s = {}
        # Initialize metrics (see EpyNN/nnlibs/commons/metrics.py)
        for s in se_config['metrics_list']:
            self.s[s] = []
            # Add list for each dataset (training, testing, validation)
            for i in range(3):
                self.s[s].append([])


class hPars:

    @log_method
    def __init__(self,se_hPars):

        # Number of training epochs
        self.i = se_hPars['training_epochs']

        ## Learning rate scheduling - Load parameters
        self.s = {}
        # Initial learning rate
        self.s['l'] = se_hPars['learning_rate']
        # Schedule mode
        self.s['s'] = se_hPars['schedule_mode']
        # Decay rate k
        self.s['k'] = se_hPars['decay_k']
        # Number of cycles n
        self.s['n'] = se_hPars['cycling_n']
        # Descent d for initial lr along cycles
        self.s['d'] = se_hPars['descent_d']

        ## Learning rate scheduling - Evaluate parameters
        # Number of epochs per cycle c
        self.s['c'] = self.i // self.s['n']
        # Default decay
        if self.s['k'] == 0:
            # 0.005% of initial lr for last epoch in cycle
            self.s['k'] = 10 / self.s['c']

        # Compute learning rate along training epochs
        self.l = cs.schedule_mode(self)

        ## Constant parameters for regularization and activation functions
        self.c = {}
        # l1 regularization
        self.c['l1'] = se_hPars['regularization_l1']
        # l2 regularization
        self.c['l2'] = se_hPars['regularization_l2']
        # Softmax temperature factor
        self.c['s'] = se_hPars['softmax_temperature']
        # Leaky ReLU alpha parameter
        self.c['l'] = se_hPars['LRELU_alpha']
        # ELU alpha parameter
        self.c['e'] = se_hPars['ELU_alpha']
        # Minimum parameter Epsilon (avoid division by zero, log of zero...)
        self.c['E'] = se_hPars['min_epsilon']

        cm.global_constant(self)
