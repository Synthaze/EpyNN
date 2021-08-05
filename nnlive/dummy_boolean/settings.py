# EpyNN/nnlive/dummy_boolean/settings.py


# GENERAL CONFIGURATION SETTINGS
config = {
    # Globals for model training
    'training_epochs': 1000,
    'training_loss': 'BCE',
    'metrics_target': 'accuracy',
    'dataset_target': 1,
    # Logs behavior
    'logs_frequency': 1,
    'logs_frequency_display': 25,
    'metrics_list': ['accuracy', 'BCE', 'MSE'],
    'metrics_plot': ['accuracy', 'BCE'],
    # Make and remove on disk
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
    'learning_rate': 0.1,
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
}
"""Hyperparameters dictionary settings.

Set hyperparameters for model and layer.
"""
