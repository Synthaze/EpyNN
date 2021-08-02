# EpyNN/nnlive/dummy_boolean/settings.py


# DATASET SETTINGS
dataset = {
    # Settings for preparation
    'N_SAMPLES': 1000,
    'dataset_name': 'dummy',
    'dataset_save': False,
    # Settings for embedding
    'dtrain_relative': 2,
    'dtest_relative': 1,
    'dval_relative': 1,
    'batch_number': 1,
}
"""Dictionary settings for dataset preparation.
"""


# GENERAL CONFIGURATION SETTINGS
config = {
    # Globals for model training
    'training_epochs': 100,
    'training_loss': 'BCE',
    'metrics_target': 'accuracy',
    'dataset_target': 1,
    # Logs behavior
    'logs_frequency': 1,
    'logs_frequency_display': 1,
    'metrics_list': ['accuracy', 'BCE'],
    'print_over': True,
    # Plot behavior
    'metrics_plot': ['accuracy', 'BCE'],
    'pyplot': True,
    'gnuplot': False,
    # Make and remove on disk
    'model_save': False,
    'directory_clear': False,
}
"""General configuration dictionary settings.

Target dataset can be picked in:
[0, 1, 2]    # [dtrain, dtest, dval]

Metrics can be picked in:
['accuracy', 'CCE', 'MSE', 'MAE', 'RMSLE']

For binary classification, extra-metrics can be picked in:
['precision', 'recall', 'BCE']
"""


# HYPERPARAMETERS SETTINGS
hPars = {
    # Schedule learning rate
    'learning_rate': 0,
    'schedule_mode': 'steady',
    'decay_k': 0.001,
    'cycling_n': 1,
    'descent_d': 1,
    # Regularization
    'regularization_l2': 0,
    'regularization_l1': 0,
    # Tune activation function
    'ELU_alpha': 0.01,
    'LRELU_alpha': 0.01,
    'softmax_temperature': 1,
    # May prevent from floating point error
    'min_epsilon': 1e-9,
}
"""Hyperparameters dictionary settings.

Set hyperparameters for model and layer.
"""
