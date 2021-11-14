# EpyNN/epynnlive/dummy_boolean/settings.py


# HYPERPARAMETERS SETTINGS
se_hPars = {
    # Schedule learning rate
    'learning_rate': 0.1,
    'schedule_mode': 'steady',
    'decay_k': 0.001,
    'cycling_n': 1,
    'descent_d': 1,
    # Tune activation function
    'ELU_alpha': 0.01,
    'LRELU_alpha': 0.01,
    'softmax_temperature': 1,
}
"""Hyperparameters dictionary settings.

Set hyperparameters for model and layer.
"""
