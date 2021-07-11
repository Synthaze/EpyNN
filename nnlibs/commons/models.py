#EpyNN/nnlibs/commons/models.py
from nnlibs.commons.decorators import *
import nnlibs.commons.schedule as cs
import nnlibs.commons.maths as cm

import numpy as np
import time


class runData:

    @log_method
    def __init__(self,se_datasets,se_config):

        datasets = se_datasets
        config = se_config

        ## Metadata
        self.m = {}
        # Experiment Name
        self.m['n'] = config['experiment_name']
        # Unique identifier
        self.m['t'] = str(int(time.time()))
        # Experiment identifier
        self.m['nt'] = self.m['n']+'_'+self.m['t']
        # Logs frequency
        self.m['f'] = config['logs_frequency']
        # Logs frequency
        self.m['fd'] = config['logs_frequency_display']
        # Target dataset
        self.m['d'] = config['dataset_target']
        # Target metrics
        self.m['m'] = config['metrics_target']
        # Metrics to plot
        self.m['p'] = config['metrics_plot']
        # Number of samples
        self.m['s'] = datasets['N_SAMPLES']

        ## Bolean for display and write on disk actions
        self.b = {}
        # Save model parameters and associated data
        self.b['ms'] = config['model_save']
        # Display plot at the end of training
        self.b['pd'] = config['plot_display']
        # Save plot at the end of training
        self.b['ps'] = config['plot_save']
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
        for s in config['metrics_list']:
            self.s[s] = []
            # Add list for each dataset (training, testing, validation)
            for i in range(3):
                self.s[s].append([])


class hPars:

    @log_method
    def __init__(self,se_hPars):

        hPars = se_hPars

        # Number of training epochs
        self.i = hPars['training_epochs']


        ## Learning rate scheduling - Load parameters
        self.s = {}
        # Initial learning rate
        self.s['l'] = hPars['learning_rate']
        # Schedule mode
        self.s['s'] = hPars['schedule_mode']
        # Decay rate k
        self.s['k'] = hPars['decay_k']
        # Number of cycles n
        self.s['n'] = hPars['cycling_n']
        # Descent d for initial lr along cycles
        self.s['d'] = hPars['descent_d']

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
        self.c['l1'] = hPars['regularization_l1']
        # l2 regularization
        self.c['l2'] = hPars['regularization_l2']
        # Softmax temperature factor
        self.c['s'] = hPars['softmax_temperature']
        # Leaky ReLU alpha parameter
        self.c['l'] = hPars['LRELU_alpha']
        # ELU alpha parameter
        self.c['e'] = hPars['ELU_alpha']
        # Minimum parameter Epsilon (avoid division by zero, log of zero...)
        self.c['E'] = hPars['min_epsilon']

        cm.global_constant(self)
