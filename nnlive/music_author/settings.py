#EpyNN/nnlibs/settings.py

config = {
    'experiment_name': 'DUMMY_WAVE',
    'logs_frequency': 1,
    'logs_frequency_display': 25,
    'model_save': False,
    'dsets_save': False,
    'hPars_save': False,
    'runData_save': False,
    'plot_display': True,
    'plot_save': False,
    'directory_clear': True,
    'dataset_target': 1,
    'metrics_target': 'accuracy',
    'N_SAMPLES': 500,
    'metrics_list': ['accuracy','recall','precision','BCE','MSE','MAE','RMSLE','KLD'],
    'metrics_plot': ['accuracy','BCE']
}

hPars = {
    'training_epochs': 1000,
    'batch_number': 1,
    'learning_rate': 0.1,
    'schedule_mode': 'exp_decay',
    'decay_k': 0.01,
    'cycling_n': 1,
    'descent_d': 1,

    'regularization_l2': 0,
    'regularization_l1': 0,

    'softmax_temperature': 1,
    'ELU_alpha': 0.01,
    'LRELU_alpha': 1,
    'min_epsilon': 1e-9,
}
